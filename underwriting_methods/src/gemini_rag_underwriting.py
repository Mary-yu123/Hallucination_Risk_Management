import json
import time
import re
import getpass
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from google import genai
from google.genai import types


# ==================================================
# 1. Gemini API 키 입력
# ==================================================

api_key = getpass.getpass("Gemini API Key 입력: ")

if not api_key:
    raise ValueError("Gemini API 키가 입력되지 않았습니다.")

client = genai.Client(api_key=api_key)


# ==================================================
# 2. 기본 설정
# ==================================================

# 대량 처리용으로 flash-lite 권장
MODEL_ID = "gemini-2.5-flash-lite"

# 고객 데이터 엑셀 파일
CUSTOMER_EXCEL_PATH = "./AutoInsuranceClaims_수식오류수정 (1).xlsx"

# 인수 가이드라인 PDF 파일
GUIDELINE_PDF_PATH = "./underwriting_guideline.pdf"

# 한 번에 몇 명씩 처리할지
# RAG는 PDF 근거도 같이 들어가므로 10~20 권장
BATCH_SIZE = 5

# API 호출 사이 대기 시간
SLEEP_SECONDS = 5

# 오류 발생 시 재시도 횟수
MAX_RETRIES = 3

# RAG에서 한 번에 넣을 PDF 근거 조각 수
TOP_K_GUIDELINE_CHUNKS = 5


MAX_GUIDELINE_CHARS_PER_BATCH = 12000


PARTIAL_OUTPUT_FILE = "rag_underwriting_results_partial최종.json"
FINAL_OUTPUT_FILE = "rag_underwriting_results최종.json"


RAW_ERROR_RESPONSE_FILE = "rag_raw_error_response최종.txt"



def clean_json_text(text):
    """
    Gemini가 ```json ... ``` 형태로 감싸거나
    앞뒤에 설명을 붙였을 경우 JSON 부분만 추출.
    """
    if text is None:
        return ""

    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()

    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    first = text.find("{")
    last = text.rfind("}")

    if first != -1 and last != -1:
        text = text[first:last + 1]

    return text

def classify_result(score):
    if score <= 25:
        return "일반 인수"
    elif score <= 55:
        return "추가 심사"
    elif score <= 85:
        return "조건부 인수 검토"
    else:
        return "인수 보류 또는 공동인수 검토"


def to_number(value):
    if value is None:
        return 0

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value) if value.is_integer() else value

    text = str(value)
    match = re.search(r"-?\d+(\.\d+)?", text)

    if match:
        number = float(match.group())
        return int(number) if number.is_integer() else number

    return 0


def extract_point(reason):
    """
    scoring_breakdown 또는 주요근거 안에서 반영점수를 추출한다.
    우선순위:
    1. reason["반영점수"]
    2. reason["적용정책"]["반영점수"]
    3. reason["적용정책"] 문자열 안 숫자
    """

    if not isinstance(reason, dict):
        return 0

    if "반영점수" in reason:
        return to_number(reason.get("반영점수"))

    policy = reason.get("적용정책", 0)

    if isinstance(policy, dict):
        if "반영점수" in policy:
            return to_number(policy.get("반영점수"))
        if "규칙" in policy:
            return to_number(policy.get("규칙"))

    return to_number(policy)


def fix_score_consistency(item):
    """
    핵심 보정 함수.
    Gemini가 위험점수를 틀리게 출력해도,
    저장 직전에 scoring_breakdown의 반영점수 합계로 위험점수를 강제 수정한다.
    """

    reasons = None

    if isinstance(item.get("scoring_breakdown"), list):
        reasons = item.get("scoring_breakdown")
    elif isinstance(item.get("주요근거"), list):
        reasons = item.get("주요근거")
    elif isinstance(item.get("데이터상품판단"), list):
        reasons = item.get("데이터상품판단")
    else:
        reasons = []

    points = []

    for reason in reasons:
        point = extract_point(reason)
        points.append(point)

        if isinstance(reason, dict):
            reason["반영점수"] = point

    fixed_score = sum(points)

    if isinstance(fixed_score, float) and fixed_score.is_integer():
        fixed_score = int(fixed_score)

    item["위험점수"] = fixed_score
    item["인수결과"] = classify_result(fixed_score)

    if points:
        item["점수검산"] = " + ".join(str(point) for point in points) + f" = {fixed_score}"
    else:
        item["점수검산"] = f"0 = {fixed_score}"

    return item
# ==================================================
# 4. PDF 읽기 및 RAG용 조각 만들기
# ==================================================

def load_pdf_chunks(pdf_path, chunk_size=2500):
    """
    PDF를 읽어서 페이지 단위 + 글자 수 기준으로 조각화.
    스캔본 PDF는 텍스트 추출이 안 될 수 있음.
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    reader = PdfReader(str(pdf_file))
    chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()

        if not page_text:
            continue

        for i in range(0, len(page_text), chunk_size):
            chunk_text = page_text[i:i + chunk_size].strip()

            if chunk_text:
                chunks.append({
                    "page": page_num,
                    "text": chunk_text
                })

    if not chunks:
        raise ValueError(
            "PDF에서 텍스트를 추출하지 못했습니다. "
            "스캔본 PDF일 가능성이 있습니다."
        )

    return chunks


def tokenize_for_search(text):
    """
    간단한 검색용 토큰화.
    한글, 영어, 숫자 단어를 추출.
    """
    text = str(text).lower()
    tokens = re.findall(r"[가-힣a-zA-Z0-9_]+", text)

    stopwords = {
        "nan", "none", "null", "true", "false",
        "and", "or", "the", "a", "an",
        "고객", "데이터", "자료", "없음"
    }

    tokens = [
        token for token in tokens
        if len(token) >= 2 and token not in stopwords
    ]

    return set(tokens)


def select_relevant_guideline_chunks(batch_df, guideline_chunks, top_k=5, max_chars=12000):
    """
    현재 고객 배치와 관련 있어 보이는 PDF 조각을 고르는 간단 RAG 검색.
    임베딩 없이 키워드 겹침 기반으로 선택.
    """
    batch_text = batch_df.astype(str).to_string(index=False)
    query_tokens = tokenize_for_search(batch_text)

    # 자동차보험 인수심사에서 중요한 일반 키워드 추가
    extra_terms = {
        "보험", "자동차", "인수", "심사", "위험", "사고", "청구",
        "claim", "claims", "vehicle", "auto", "insurance",
        "coverage", "risk", "premium", "policy", "underwriting"
    }

    query_tokens = query_tokens.union(extra_terms)

    scored_chunks = []

    for chunk in guideline_chunks:
        chunk_text_lower = chunk["text"].lower()

        score = 0
        for token in query_tokens:
            if token.lower() in chunk_text_lower:
                score += 1

        scored_chunks.append({
            "score": score,
            "page": chunk["page"],
            "text": chunk["text"]
        })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    selected = []

    for item in scored_chunks:
        if len(selected) >= top_k:
            break

        # 점수가 0이어도 fallback으로 앞부분 몇 개는 넣음
        selected.append(item)

    rag_text_parts = []
    total_chars = 0

    for item in selected:
        part = f"[PDF page {item['page']} / relevance score {item['score']}]\n{item['text']}"
        part_len = len(part)

        if total_chars + part_len > max_chars:
            remaining = max_chars - total_chars

            if remaining > 500:
                rag_text_parts.append(part[:remaining])
            break

        rag_text_parts.append(part)
        total_chars += part_len

    return "\n\n---\n\n".join(rag_text_parts)


# ==================================================
# 5. 이전 결과 이어서 읽기
# ==================================================

def load_previous_results():
    """
    중간 저장 파일이 있으면 이어서 실행.
    """
    path = Path(PARTIAL_OUTPUT_FILE)

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            results = data.get("results", [])
            print(f"기존 RAG 중간 저장 파일 발견: {len(results)}명 처리된 상태에서 이어서 시작합니다.")
            return results

        except Exception:
            print("중간 저장 파일을 읽지 못했습니다. 처음부터 다시 시작합니다.")
            return []

    return []


def save_partial_results(results):
    """
    중간 결과 저장.
    """
    partial_output = {
        "results": results
    }

    with open(PARTIAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(partial_output, f, ensure_ascii=False, indent=2)

    print(f"중간 저장 완료: {PARTIAL_OUTPUT_FILE}")


# ==================================================
# 6. 엑셀 파일 읽기
# ==================================================

try:
    customer_df = pd.read_excel(CUSTOMER_EXCEL_PATH)
    print("고객 데이터가 성공적으로 로드되었습니다.")
    print("전체 고객 수:", len(customer_df))
    print(customer_df.head().to_string())

except FileNotFoundError:
    print("오류: 고객 엑셀 파일을 찾을 수 없습니다.")
    print(f"현재 설정된 엑셀 파일명: {CUSTOMER_EXCEL_PATH}")
    print("엑셀 파일이 이 파이썬 파일과 같은 폴더에 있는지 확인하세요.")
    raise


# Customer 컬럼이 없으면 자동 생성
if "Customer" not in customer_df.columns:
    customer_df.insert(
        0,
        "Customer",
        [f"Customer_{i + 1}" for i in range(len(customer_df))]
    )


# ==================================================
# 7. PDF 가이드라인 읽기
# ==================================================

try:
    guideline_chunks = load_pdf_chunks(GUIDELINE_PDF_PATH)
    print("인수 가이드라인 PDF가 성공적으로 로드되었습니다.")
    print("PDF 조각 수:", len(guideline_chunks))

except FileNotFoundError:
    print("오류: 인수 가이드라인 PDF를 찾을 수 없습니다.")
    print(f"현재 설정된 PDF 파일명: {GUIDELINE_PDF_PATH}")
    print("PDF 파일이 이 파이썬 파일과 같은 폴더에 있는지 확인하세요.")
    raise


# ==================================================
# 8. 이전 결과 이어서 실행
# ==================================================

all_results = load_previous_results()

# 기존 partial 파일에 저장된 결과도 위험점수와 점수검산을 다시 맞춘다.
all_results = [fix_score_consistency(item) for item in all_results]

already_done = len(all_results)

if already_done >= len(customer_df):
    print("이미 모든 고객 처리가 완료되어 있습니다.")
else:
    print(f"이번 실행 시작 위치: {already_done + 1}번째 고객")


# ==================================================
# 9. 배치별 RAG 기반 Gemini API 호출
# ==================================================

for start in range(already_done, len(customer_df), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch_df = customer_df.iloc[start:end]

    print(f"\n처리 중: {start + 1}번째 고객부터 {min(end, len(customer_df))}번째 고객까지")

    customer_data_json = batch_df.to_json(orient="records", force_ascii=False)

    # 현재 고객 배치와 관련된 PDF 근거만 선택
    guideline_context = select_relevant_guideline_chunks(
        batch_df=batch_df,
        guideline_chunks=guideline_chunks,
        top_k=TOP_K_GUIDELINE_CHUNKS,
        max_chars=MAX_GUIDELINE_CHARS_PER_BATCH
    )

    prompt = f"""
너는 자동차보험 인수심사 AI다.
이번 작업은 RAG 기반 고객 인수심사다.
아래 PDF 인수 가이드라인 근거와 고객 데이터를 함께 사용하여 인수심사 결과 초안을 작성하라.

핵심 원칙:
- PDF 가이드라인에 명시된 기준만 점수 산정에 사용하라.
- PDF에 명시되지 않은 항목은 위험점수에 반영하지 말고 "자료 없음" 또는 "가이드 미명시"로 처리하라.
- 일반 보험 지식은 근거 설명 보완에만 사용하고, 점수 산정에는 사용하지 마라.
- 존재하지 않는 고객 데이터는 만들지 말고 "자료 없음"으로 처리하라.
- Gender, Education, Marital Status, Income, Employment Status는 직접 인수거절 또는 보험료 할증 근거로 쓰지 않는다.
- 음주운전, 법규위반, 사고횟수, 차량연식, 차량가액이 고객 데이터에 없으면 자료 없음으로 처리한다.
-위험점수는 각 데이터상품판단의 반영점수 합계로 산출하라.
- 반영점수는 반드시 적용정책에 명시하라.
- 위험점수와 반영점수 합계가 다르면 안 된다.
- 최종 인수 여부는 언더라이터가 판단한다.
- 각 고객에 대해 PDF 가이드라인의 모든 스코어링 항목을 검토하라.

점수 분류 기준:
- 0~25점: 일반 인수
- 26~55점: 추가 심사
- 56~85점: 조건부 인수 검토
- 86점 이상: 인수 보류 또는 공동인수 검토

출력 조건:
1. 자연어 설명 없이 JSON만 출력할 것
2. 반드시 최상위 객체는 "results" 키를 가질 것
3. 반드시 고객 식별자 Customer를 포함할 것
4. 위험점수는 숫자 형태로 출력할 것
5. 인수결과는 문자열 형태로 출력할 것
6. 각 고객에 대해 PDF 가이드라인의 모든 스코어링 항목을 검토할 것
7. 해당되는 모든 규칙을 scoring_breakdown에 나열하라.
9. scoring_breakdown의 각 항목에는 판단기준, 입력데이터, 적용정책, 근거설명, 반영점수, 참조가이드라인을 포함할 것
10. 하나의 근거만 선택하지 말고, 점수에 반영된 모든 근거를 출력할 것
11. JSON 문법을 반드시 지킬 것
12. 마지막 원소 뒤에 쉼표를 붙이지 말 것
13.  중복 설명 금지

반드시 아래 구조로만 출력하라.

{{
  "results": [
    {{
      "Customer": "Customer_1",
      "위험점수": 65,
      "인수결과": "조건부 인수 검토",
      "scoring_breakdown": [
        {{
          "판단기준": "미해결 민원 3건 이상",
          "입력데이터": "Number of Open Complaints: 3",
          "적용정책": "40점 가산",
          "근거설명": "미해결 민원 3건 이상으로 40점 반영",
          "반영점수": 40,
          "참조가이드라인": "PDF page 3"
        }},
        {{
          "판단기준": "추가 해당 조건",
          "입력데이터": "실제 고객 데이터 값",
          "적용정책": "25점 가산",
          "근거설명": "PDF 기준에 따라 25점 추가 반영",
          "반영점수": 25,
          "참조가이드라인": "PDF page 2"
        }}
      ],
      "점수검산": "40 + 25 = 65"
    }}
  ]
}}
   
--- RAG 검색으로 선택된 PDF 인수 가이드라인 근거 ---
{guideline_context}
---

--- 고객 데이터 ---
{customer_data_json}
---
"""

    success = False

            
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Gemini 호출 시도 {attempt}/{MAX_RETRIES}")
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=20000
                )
            )

            result_text = response.text
            result_text = clean_json_text(result_text)

            try:
                result_json = json.loads(result_text)

            except json.JSONDecodeError as json_error:
                with open(RAW_ERROR_RESPONSE_FILE, "w", encoding="utf-8") as f:
                    f.write(result_text)

                print("JSON 파싱 오류가 발생했습니다.")
                print(json_error)
                print(f"깨진 응답 원문 저장: {RAW_ERROR_RESPONSE_FILE}")

                if attempt < MAX_RETRIES:
                    print("같은 배치를 다시 시도합니다.")
                    time.sleep(10)
                    continue

                raise

            batch_results = result_json.get("results", [])

            if not batch_results:
                raise ValueError("Gemini 응답에 results가 없습니다.")

            fixed_batch_results = [
                fix_score_consistency(item)
                for item in batch_results
            ]

            all_results.extend(fixed_batch_results)

            all_results = [
                fix_score_consistency(item)
                for item in all_results
            ]

            print(f"완료: 이번 배치 {len(fixed_batch_results)}명 처리")
            print(f"누적 처리 고객 수: {len(all_results)}명")

            save_partial_results(all_results)

            success = True
            break

        except Exception as e:
            error_message = str(e)

            print("\nGemini API 호출 또는 처리 중 오류가 발생했습니다.")
            print(error_message)

            if "API_KEY_INVALID" in error_message or "API key not valid" in error_message:
                print("\n원인: Gemini API 키가 잘못되었습니다.")
                print("해결: AI Studio에서 복사한 AIza... 형식의 Gemini API 키를 다시 입력하세요.")
                raise

            if "RESOURCE_EXHAUSTED" in error_message or "Quota exceeded" in error_message:
                print("\n원인: Gemini API 할당량 또는 요청 한도 초과입니다.")
                print("해결: 잠시 기다리거나, 결제/한도 설정을 확인하세요.")
                raise

            if "503" in error_message or "UNAVAILABLE" in error_message:
                wait_time = 30 * attempt
                print(f"\n원인: Gemini 서버 혼잡입니다. {wait_time}초 후 재시도합니다.")
                time.sleep(wait_time)
                continue

            if attempt < MAX_RETRIES:
                wait_time = 10 * attempt
                print(f"{wait_time}초 후 재시도합니다.")
                time.sleep(wait_time)
                continue

            print(f"오류 발생 위치: {start + 1}번째 고객부터 {min(end, len(customer_df))}번째 고객까지")
            print("해결 방법:")
            print("1. BATCH_SIZE를 10에서 5로 줄이세요.")
            print("2. 다시 실행하면 중간 저장 파일 기준으로 이어서 처리됩니다.")
            raise
    if not success:
        print("최대 재시도 횟수를 초과했습니다.")
        raise RuntimeError("배치 처리 실패")

    time.sleep(SLEEP_SECONDS)


# ==================================================
# 10. 최종 결과 저장
# ==================================================
# ==================================================
# 10. 최종 결과 저장
# ==================================================

# 최종 저장 전에도 위험점수와 점수검산을 다시 한 번 강제 일치시킨다.
all_results = [
    fix_score_consistency(item)
    for item in all_results
]

final_output = {
    "results": all_results
}

with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

print("\n전체 완료.")
print(f"최종 결과 파일: {FINAL_OUTPUT_FILE}")
print(f"중간 저장 파일: {PARTIAL_OUTPUT_FILE}")
print("총 처리 고객 수:", len(all_results))