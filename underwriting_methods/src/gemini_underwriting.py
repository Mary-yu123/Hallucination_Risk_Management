import json
import time
import getpass
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types


# --------------------------------------------------
# 1. Gemini API 키 입력
# --------------------------------------------------
# 실행하면 "Gemini API Key 입력:" 이 뜹니다.
# 그때 Gemini API 키를 붙여넣고 Enter를 누르세요.
# 입력해도 화면에 안 보이는 게 정상입니다.

api_key = getpass.getpass("Gemini API Key 입력: ")

if not api_key:
    raise ValueError("Gemini API 키가 입력되지 않았습니다.")

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# 2. 기본 설정
# --------------------------------------------------
MODEL_ID = "gemini-2.5-flash-lite"
# 엑셀 파일 이름
file_path = "./AutoInsuranceClaims_수식오류수정 (1).xlsx"

# 9000개 전체를 한 번에 보내지 않고 20개씩 나눠서 처리
# 오류가 나면 20을 10 또는 5로 줄이세요.
BATCH_SIZE = 20

# 무료 등급에서 너무 빨리 호출하면 막힐 수 있어서 배치 사이에 잠깐 쉼
SLEEP_SECONDS = 5

# 저장 파일
PARTIAL_OUTPUT_FILE = "gemini_underwriting_results_partial.json"
FINAL_OUTPUT_FILE = "gemini_underwriting_results.json"


# --------------------------------------------------
# 3. JSON 파싱 보조 함수
# --------------------------------------------------

def clean_json_text(text):
    """
    Gemini가 혹시 ```json ... ``` 형태로 감싸서 줄 경우 제거.
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()

    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    # 앞뒤에 쓸데없는 문장이 붙었을 경우 JSON 부분만 추출
    first = text.find("{")
    last = text.rfind("}")

    if first != -1 and last != -1:
        text = text[first:last + 1]

    return text


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
            print(f"기존 중간 저장 파일 발견: {len(results)}명 처리된 상태에서 이어서 시작합니다.")
            return results

        except Exception:
            print("중간 저장 파일을 읽지 못했습니다. 처음부터 다시 시작합니다.")
            return []

    return []


# --------------------------------------------------
# 4. 엑셀 파일 읽기
# --------------------------------------------------

try:
    customer_df = pd.read_excel(file_path)
    print("고객 데이터가 성공적으로 로드되었습니다.")
    print("전체 고객 수:", len(customer_df))
    print(customer_df.head().to_string())

except FileNotFoundError:
    print("오류: 엑셀 파일을 찾을 수 없습니다.")
    print("엑셀 파일이 gemini_underwriting.py와 같은 폴더에 있는지 확인하세요.")
    raise


# Customer 컬럼이 없으면 자동 생성
if "Customer" not in customer_df.columns:
    customer_df.insert(0, "Customer", [f"Customer_{i + 1}" for i in range(len(customer_df))])


# --------------------------------------------------
# 5. 이전 결과가 있으면 이어서 실행
# --------------------------------------------------

all_results = load_previous_results()
already_done = len(all_results)

if already_done >= len(customer_df):
    print("이미 모든 고객 처리가 완료되어 있습니다.")
else:
    print(f"이번 실행 시작 위치: {already_done + 1}번째 고객")


# --------------------------------------------------
# 6. 배치별 Gemini API 호출
# --------------------------------------------------

for start in range(already_done, len(customer_df), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch_df = customer_df.iloc[start:end]

    print(f"\n처리 중: {start + 1}번째 고객부터 {min(end, len(customer_df))}번째 고객까지")

    customer_data_json = batch_df.to_json(orient="records", force_ascii=False)

    prompt = f"""
너는 자동차보험 인수심사 AI다.
아래 고객 데이터에 대해 인수심사 결과 초안을 작성하라.

중요 조건:
- 상세 인수심사 스코어링 기준표는 제공되지 않는다.
- 너의 보험 인수심사 지식을 바탕으로 각 변수의 위험도를 판단하라.
- 음주운전, 법규위반, 사고횟수, 차량연식, 차량가액은 자료 없음으로 처리한다.
- Gender, Education, Marital Status, Income, Employment Status는 직접 인수거절 또는 보험료 할증 근거로 쓰지 않는다.
- 최종 인수 여부는 언더라이터가 판단한다.
- 존재하지 않는 데이터는 생성하지 말고, 자료 없음으로 처리하라.

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
6. 주요근거는 리스트 형태로 출력할 것
7. 각 주요근거에는 반드시 아래 항목을 포함할 것:
   - 판단기준
   - 입력데이터
   - 적용정책
   - 근거설명
8. 입력데이터에는 실제 사용한 고객 데이터 값을 그대로 작성할 것
9. 적용정책에는 적용한 인수심사 규칙과 반영한 위험점수 또는 정책 결과를 포함할 것
10. 모든 판단은 실제 입력 데이터와 위 기준에 근거해야 함

반드시 아래 JSON 구조로만 출력하라.

{{
  "results": [
    {{
      "Customer": "Customer_1",
      "위험점수": 0,
      "인수결과": "일반 인수",
      "주요근거": [
        {{
          "판단기준": "판단 기준",
          "입력데이터": "실제 입력 데이터",
          "적용정책": "적용한 규칙 및 반영 점수",
          "근거설명": "입력 데이터, 적용 정책, 최종 판단을 연결한 설명"
        }}
      ]
    }}
  ]
}}

--- 고객 데이터 ---
{customer_data_json}
---
"""

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=30000
            )
        )

        result_text = response.text
        result_text = clean_json_text(result_text)

        result_json = json.loads(result_text)

        batch_results = result_json.get("results", [])

        if not batch_results:
            raise ValueError("Gemini 응답에 results가 없습니다.")

        all_results.extend(batch_results)

        print(f"완료: 이번 배치 {len(batch_results)}명 처리")
        print(f"누적 처리 고객 수: {len(all_results)}명")

        # 중간 저장
        partial_output = {
            "results": all_results
        }

        with open(PARTIAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(partial_output, f, ensure_ascii=False, indent=2)

        print(f"중간 저장 완료: {PARTIAL_OUTPUT_FILE}")

        time.sleep(SLEEP_SECONDS)

    except Exception as e:
        print("\nGemini API 호출 중 오류가 발생했습니다.")
        print(e)
        print(f"오류 발생 위치: {start + 1}번째 고객부터 {min(end, len(customer_df))}번째 고객까지")
        print("\n해결 방법:")
        print("1. 무료 할당량 오류면 잠시 기다렸다가 다시 실행하세요.")
        print("2. 입력/출력 길이 오류면 BATCH_SIZE = 20을 10 또는 5로 줄이세요.")
        print("3. 다시 실행하면 중간 저장 파일 기준으로 이어서 처리됩니다.")
        raise


# --------------------------------------------------
# 7. 최종 결과 저장
# --------------------------------------------------

final_output = {
    "results": all_results
}

with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

print("\n전체 완료.")
print(f"최종 결과 파일: {FINAL_OUTPUT_FILE}")
print(f"중간 저장 파일: {PARTIAL_OUTPUT_FILE}")
print("총 처리 고객 수:", len(all_results))