import json
import time
import re
import getpass
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types


API_KEY = getpass.getpass("Gemini API Key 입력: ")

if not API_KEY:
    raise ValueError("Gemini API 키가 입력되지 않았습니다.")

client = genai.Client(api_key=API_KEY)


MODEL_ID = "gemini-2.5-flash-lite"

EXCEL_PATH = "./AutoInsuranceClaims_mesh.xlsx"

DATA_PRODUCT_SHEETS = ["고객", "계약", "청구민원", "자동차", "crm"]
BASE_SHEET_NAME = "고객"

BATCH_SIZE = 5AIzaSyBjqH6ZQ1MnRDL6LtDYon209sMBLcOcL9s
SLEEP_SECONDS = 5
MAX_RETRIES = 3

PARTIAL_OUTPUT_FILE = "gemini_mesh_rag_experiment2_results_partial.json"
FINAL_OUTPUT_FILE = "gemini_mesh_rag_experiment2_results.json"
VALIDATION_JSON_FILE = "gemini_mesh_rag_experiment2_validation.json"
VALIDATION_XLSX_FILE = "gemini_mesh_rag_experiment2_validation.xlsx"
RAW_ERROR_RESPONSE_FILE = "gemini_mesh_rag_experiment2_raw_error_response.txt"

GUIDELINE_RULES = """
자동차보험 인수심사 가이드라인 스코어링 기준:

[3-1-A] 최근 청구 이력 / 청구민원 데이터 상품 / Months Since Last Claim
- 24개월 이상: 0점
- 12개월 이상 24개월 미만: 5점
- 6개월 이상 12개월 미만: 15점
- 6개월 미만: 25점

[3-1-B] 총 청구금액 / 청구민원 데이터 상품 / Total Claim Amount
- $518 이하: 0점
- $518 초과 ~ $739 이하: 10점
- $739 초과 ~ $1,044 이하: 20점
- $1,044 초과: 35점

[3-1-C] 미해결 민원 / 청구민원 데이터 상품 / Number of Open Complaints
- 0건: 0점
- 1건: 10점
- 2건: 25점
- 3건 이상: 40점

[3-2-A] 차량 등급 / 자동차 데이터 상품 / Vehicle Class
- Sports Car: 25점
- Luxury Car: 25점
- Luxury SUV: 25점
- SUV: 10점
- Four-Door Car: 0점
- Two-Door Car: 0점
- 값 없음 또는 분류 불명확: 10점

[3-2-B] 차량 크기 / 자동차 데이터 상품 / Vehicle Size
- Large: 10점
- Medsize: 0점
- Small: 0점

[3-3-A] 계약 경과 기간 / 계약 데이터 상품 / Months Since Policy Inception
- 6개월 미만: 10점
- 6개월 이상 12개월 미만: 5점
- 12개월 이상: 0점

[3-4-B] 거주 지역 유형 / 고객 데이터 상품 / Location
- Suburban: 5점
- Urban: 0점
- Rural: 0점

[Rule 1] 고위험 차종 + 최근 청구 / 자동차 + 청구민원 데이터 상품
- 조건: Vehicle Class ∈ {Sports Car, Luxury Car, Luxury SUV}
  AND Months Since Last Claim < 12
- 반영점수: 20점

[Rule 2] 고액 청구 + 민원 / 청구민원 데이터 상품
- 조건: Total Claim Amount ≥ 739
  AND Number of Open Complaints ≥ 2
- 반영점수: 15점

[Rule 3] 신규 + 고액 청구 / 계약 + 청구민원 데이터 상품
- 조건: Months Since Policy Inception < 6
  AND Total Claim Amount ≥ 1044
- 반영점수: 10점

복합 리스크 규칙:
- Rule 1, Rule 2, Rule 3은 중복 적용 가능하다.
- 단, 동일 조건을 두 번 가산하지 않는다.

점수 분류 기준:
- 0~25점: 일반 인수
- 26~55점: 추가 심사
- 56~85점: 조건부 인수 검토
- 86점 이상: 인수 보류 또는 공동인수 검토

Data Mesh 해석:
- 고객 sheet: Location 기준 판단에 사용
- 계약 sheet: Months Since Policy Inception 기준 판단 및 Rule 3 일부 판단에 사용
- 청구민원 sheet: Months Since Last Claim, Total Claim Amount, Number of Open Complaints, Rule 1, Rule 2, Rule 3 판단에 사용
- 자동차 sheet: Vehicle Class, Vehicle Size, Rule 1 일부 판단에 사용
- crm sheet: PDF 점수표에 직접 대응되는 기준 없음. 점수 산정에는 사용하지 않음.
"""


def clean_json_text(text):
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


def normalize_column_name(name):
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def clean_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def find_customer_column(df):
    candidates = [
        "Customer",
        "customer",
        "CUSTOMER",
        "Customer_ID",
        "Customer ID",
        "customer_id",
        "ID",
        "id",
        "고객ID",
        "고객 ID",
        "고객번호",
        "고객 번호"
    ]

    col_map = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in col_map:
            return col_map[key]

    return None


def load_data_products():
    excel_file = Path(EXCEL_PATH)

    if not excel_file.exists():
        raise FileNotFoundError(f"엑셀 파일을 찾을 수 없습니다: {EXCEL_PATH}")

    sheets = pd.read_excel(excel_file, sheet_name=None)
    data_products = {}

    print("엑셀 sheet 읽는 중:")

    for sheet_name in DATA_PRODUCT_SHEETS:
        if sheet_name not in sheets:
            print(f"- {sheet_name}: 없음, 제외")
            continue

        df = sheets[sheet_name].copy()
        df.columns = [str(col).strip() for col in df.columns]
        df = df.dropna(how="all")

        if df.empty:
            print(f"- {sheet_name}: 비어 있음, 제외")
            continue

        customer_col = find_customer_column(df)

        if customer_col is None:
            raise ValueError(f"'{sheet_name}' sheet에 Customer 컬럼이 없습니다.")

        if customer_col != "Customer":
            df = df.rename(columns={customer_col: "Customer"})

        df["Customer"] = df["Customer"].astype(str).str.strip()
        data_products[sheet_name] = df

        print(f"- {sheet_name}: 사용, 행 수 {len(df)}")

    if BASE_SHEET_NAME not in data_products:
        raise ValueError("'고객' sheet가 없습니다. 고객 순서 기준 sheet가 필요합니다.")

    return data_products


def row_to_dict(row):
    result = {}

    for key, value in row.items():
        value = clean_value(value)

        if value is not None:
            result[str(key).strip()] = value

    return result


def build_customer_payloads(data_products):
    base_df = data_products[BASE_SHEET_NAME].copy()
    base_df["Customer"] = base_df["Customer"].astype(str).str.strip()

    ordered_customers = []
    seen = set()

    for customer in base_df["Customer"].tolist():
        if customer not in seen:
            ordered_customers.append(customer)
            seen.add(customer)

    payloads = []

    for row_order, customer in enumerate(ordered_customers, start=1):
        payload = {
            "row_order": row_order,
            "Customer": customer,
            "data_products": {}
        }

        for sheet_name in DATA_PRODUCT_SHEETS:
            if sheet_name not in data_products:
                payload["data_products"][sheet_name] = []
                continue

            df = data_products[sheet_name]
            rows = df[df["Customer"].astype(str).str.strip() == customer]

            if rows.empty:
                payload["data_products"][sheet_name] = []
            else:
                payload["data_products"][sheet_name] = [
                    row_to_dict(row)
                    for _, row in rows.iterrows()
                ]

        payloads.append(payload)

    return payloads


def load_previous_results():
    path = Path(PARTIAL_OUTPUT_FILE)

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", [])
        print(f"기존 중간 저장 파일 발견: {len(results)}명 처리됨")
        return results

    return []


def save_partial_results(results):
    results = sorted(results, key=lambda x: int(x.get("row_order", 10**12)))

    with open(PARTIAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    print(f"중간 저장 완료: {PARTIAL_OUTPUT_FILE}")


def make_prompt(batch_payload):
    customer_data_json = json.dumps(batch_payload, ensure_ascii=False)

    prompt = f"""
너는 자동차보험 인수심사 AI다.
이번 실험은 RAG + Data Mesh 기반 자동차보험 인수심사 실험이다.

목표:
- 인수심사 가이드라인에 근거하여 고객 위험점수를 산출한다.
- 엑셀의 각 sheet는 데이터 상품 메타데이터로 간주한다.
- 각 고객에 대해 아래 가이드라인의 모든 기준을 내부적으로 검토한다.
- 7개 기본 기준과 복합 리스크 Rule 1, Rule 2, Rule 3을 모두 검토한다.
- 단, 출력 JSON의 주요근거에는 반영점수가 0보다 큰 항목만 포함한다.
- 조건 미충족으로 0점인 기준은 출력하지 않는다.
- 위험점수는 주요근거에 출력된 모든 적용정책.반영점수의 합계여야 한다.
- 복합 리스크 규칙은 중복 적용 가능하나, 동일 조건은 두 번 가산하지 않는다.

{GUIDELINE_RULES}

출력 조건:
1. 자연어 설명 없이 JSON만 출력할 것
2. 반드시 최상위 객체는 "results" 키를 가질 것
3. 반드시 고객 식별자 Customer를 포함할 것
4. 위험점수는 숫자 형태로 출력할 것
5. 인수결과는 문자열 형태로 출력할 것
6. 주요근거는 리스트 형태로 출력할 것
7. 주요근거에는 반영점수가 0보다 큰 기준만 포함할 것
8. 해당 고객에게 반영점수가 0보다 큰 기준이 없으면 주요근거는 빈 리스트 []로 출력할 것
9. 각 주요근거에는 반드시 아래 항목을 포함할 것:
   - 기준ID
   - 판단기준
   - 사용데이터상품
   - 입력데이터
   - 적용정책
   - 근거설명
   - 참조가이드라인
10. 입력데이터에는 실제 사용한 고객 데이터 값을 그대로 작성할 것
11. 적용정책에는 반드시 아래 항목을 포함할 것:
   - 규칙
   - 반영점수
12. 모든 판단은 반드시 실제 입력 데이터와 인수심사 기준에 근거해야 함
13. 존재하지 않는 데이터나 정책을 생성하지 말 것
14. 근거설명은 반드시 입력 데이터, 적용 정책, 최종 판단이 연결되도록 작성할 것
15. Employment Status, Income, Gender, Education, Marital Status는 점수 산정 근거로 사용하지 말 것
16. crm sheet는 데이터 상품으로 존재하지만 PDF 점수 기준에 직접 대응되는 항목이 없으므로 점수 산정에 사용하지 말 것
17. JSON 문법을 반드시 지킬 것
18. 마지막 원소 뒤에 쉼표를 붙이지 말 것

중요:
- 위험점수와 점수검산이 다르면 안 된다.
- 위험점수는 반드시 주요근거의 반영점수 합계와 같아야 한다.
- 인수결과는 위험점수 기준표에 따라 산출하라.

반드시 아래 JSON 구조로만 출력하라.

{{
  "results": [
    {{
      "row_order": 1,
      "Customer": "QC35222",
      "위험점수": 65,
      "인수결과": "조건부 인수 검토",
      "주요근거": [
        {{
          "기준ID": "3-4-B",
          "판단기준": "Location",
          "사용데이터상품": ["고객"],
          "입력데이터": {{
            "Location": "Suburban"
          }},
          "적용정책": {{
            "규칙": "Suburban → 5점",
            "반영점수": 5
          }},
          "근거설명": "고객 데이터 상품의 Location 값이 Suburban이므로 5점을 반영함",
          "참조가이드라인": "PDF page 5"
        }},
        {{
          "기준ID": "Rule 1",
          "판단기준": "고위험 차종 + 최근 청구",
          "사용데이터상품": ["자동차", "청구민원"],
          "입력데이터": {{
            "Vehicle Class": "Luxury SUV",
            "Months Since Last Claim": 4
          }},
          "적용정책": {{
            "규칙": "Vehicle Class ∈ {{Sports Car, Luxury Car, Luxury SUV}} AND Months Since Last Claim < 12 → 20점",
            "반영점수": 20
          }},
          "근거설명": "자동차와 청구민원 데이터 상품의 조건이 동시에 충족되어 Rule 1 점수를 반영함",
          "참조가이드라인": "PDF page 6"
        }}
      ],
      "점수검산": "5 + 20 + 40 = 65"
    }}
  ]
}}

--- 고객별 Data Mesh 데이터 상품 ---
{customer_data_json}
---
"""
    return prompt



def call_gemini(prompt):
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
            max_output_tokens=30000
        )
    )

    result_text = clean_json_text(response.text)
    return json.loads(result_text)


def to_number(value):
    if value is None:
        return 0

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value) if value.is_integer() else value

    text = str(value)
    match = re.search(r"-?\\d+(\\.\\d+)?", text)

    if match:
        n = float(match.group())
        return int(n) if n.is_integer() else n

    return 0
def classify_result(score):
    if score <= 25:
        return "일반 인수"
    elif score <= 55:
        return "추가 심사"
    elif score <= 85:
        return "조건부 인수 검토"
    else:
        return "인수 보류 또는 공동인수 검토"


def force_score_from_reasons(item, source_payload=None):
    """
    핵심 보정 함수.
    Gemini가 위험점수를 틀리게 출력해도,
    저장 직전에 주요근거의 반영점수 합계로 위험점수를 강제 수정한다.
    또한 0점 주요근거는 제거한다.
    """
    reasons = item.get("주요근거", [])

    if not isinstance(reasons, list):
        reasons = []

    fixed_reasons = []
    points = []

    for reason in reasons:
        if not isinstance(reason, dict):
            continue

        policy = reason.get("적용정책", {})

        if not isinstance(policy, dict):
            policy = {}

        point = to_number(policy.get("반영점수", 0))

        # 0점 기준은 최종 JSON에서 제거
        if point == 0:
            continue

        policy["반영점수"] = point

        if "규칙" not in policy:
            policy["규칙"] = f"{point}점 반영"

        reason["적용정책"] = policy

        fixed_reasons.append(reason)
        points.append(point)

    total_score = sum(points)

    if isinstance(total_score, float) and total_score.is_integer():
        total_score = int(total_score)

    if source_payload is not None:
        item["row_order"] = source_payload["row_order"]
        item["Customer"] = source_payload["Customer"]

    item["주요근거"] = fixed_reasons
    item["위험점수"] = total_score
    item["인수결과"] = classify_result(total_score)

    if points:
        item["점수검산"] = " + ".join(str(p) for p in points) + f" = {total_score}"
    else:
        item["점수검산"] = f"0 = {total_score}"

    return item

def validate_result_item(item):
    reasons = item.get("주요근거", [])

    reason_sum = 0

    for reason in reasons:
        if not isinstance(reason, dict):
            continue

        policy = reason.get("적용정책", {})

        if not isinstance(policy, dict):
            continue

        point = to_number(policy.get("반영점수", 0))
        reason_sum += point

    actual_score = to_number(item.get("위험점수", 0))

    return {
        "Customer": item.get("Customer"),
        "위험점수": actual_score,
        "주요근거_합계": reason_sum,
        "합산일치": actual_score == reason_sum,
        "주요근거개수": len(reasons),
        "점수검산": item.get("점수검산", "")
    }

def save_validation_report(results):
    validation_rows = []

    for item in results:
        validation_rows.append(validate_result_item(item))

    with open(VALIDATION_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump({"validation": validation_rows}, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(validation_rows)
    df.to_excel(VALIDATION_XLSX_FILE, index=False)

    bad_sum = df[df["합산일치"] == False]

    print("\n검증 완료")
    print("검증 JSON:", VALIDATION_JSON_FILE)
    print("검증 Excel:", VALIDATION_XLSX_FILE)
    print("합산 불일치 개수:", len(bad_sum))

def main():
    data_products = load_data_products()
    payloads = build_customer_payloads(data_products)

    previous_results = load_previous_results()
    done_customers = set(str(item.get("Customer")) for item in previous_results)

    remaining_payloads = [
        payload for payload in payloads
        if str(payload["Customer"]) not in done_customers
    ]

    all_results = previous_results[:]

    print("전체 고객 수:", len(payloads))
    print("이미 처리된 고객 수:", len(done_customers))
    print("남은 고객 수:", len(remaining_payloads))

    for start in range(0, len(remaining_payloads), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_payload = remaining_payloads[start:end]

        print(f"\n처리 중: 남은 고객 기준 {start + 1}번째부터 {min(end, len(remaining_payloads))}번째까지")

        prompt = make_prompt(batch_payload)

        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"Gemini 호출 시도 {attempt}/{MAX_RETRIES}")

                result_json = call_gemini(prompt)

                raw_batch_results = result_json.get("results", [])

                if not isinstance(raw_batch_results, list) or len(raw_batch_results) == 0:
                    raise ValueError("Gemini 응답에 results가 없거나 비어 있습니다.")

                by_customer = {}

                for item in raw_batch_results:
                    if isinstance(item, dict) and "Customer" in item:
                        by_customer[str(item["Customer"])] = item

                fixed_batch_results = []

                for idx, payload in enumerate(batch_payload):
                    customer = str(payload["Customer"])

                    if customer in by_customer:
                        item = by_customer[customer]
                    elif idx < len(raw_batch_results) and isinstance(raw_batch_results[idx], dict):
                        item = raw_batch_results[idx]
                    else:
                        item = {
                            "Customer": customer,
                            "주요근거": []
                        }

                    fixed_item = force_score_from_reasons(item, payload)
                    fixed_batch_results.append(fixed_item)

                all_results.extend(fixed_batch_results)
                all_results = sorted(all_results, key=lambda x: int(x.get("row_order", 10**12)))

                print(f"완료: 이번 배치 {len(fixed_batch_results)}명 처리")
                print(f"누적 처리 고객 수: {len(all_results)}명")

                save_partial_results(all_results)

                success = True
                break

            except json.JSONDecodeError as e:
                print("JSON 파싱 오류 발생")
                print(e)

                with open(RAW_ERROR_RESPONSE_FILE, "w", encoding="utf-8") as f:
                    f.write(str(e))

                if attempt < MAX_RETRIES:
                    time.sleep(10)
                    continue

                raise

            except Exception as e:
                error_message = str(e)
                print("Gemini 호출 또는 처리 오류")
                print(error_message)

                if "API_KEY_INVALID" in error_message or "API key not valid" in error_message:
                    raise

                if "RESOURCE_EXHAUSTED" in error_message or "Quota exceeded" in error_message:
                    raise

                if "503" in error_message or "UNAVAILABLE" in error_message:
                    wait_time = 30 * attempt
                    print(f"서버 혼잡. {wait_time}초 후 재시도")
                    time.sleep(wait_time)
                    continue

                if attempt < MAX_RETRIES:
                    time.sleep(10)
                    continue

                raise

        if not success:
            raise RuntimeError("배치 처리 실패")

        time.sleep(SLEEP_SECONDS)

    all_results = sorted(all_results, key=lambda x: int(x.get("row_order", 10**12)))

    with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, ensure_ascii=False, indent=2)

    save_validation_report(all_results)

    print("\n전체 완료")
    print("최종 결과 파일:", FINAL_OUTPUT_FILE)
    print("중간 저장 파일:", PARTIAL_OUTPUT_FILE)


main()