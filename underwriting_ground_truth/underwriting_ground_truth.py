"""
가상 자동차보험 인수심사 기준 v2.2 - 정답지 생성 스크립트
출력 1: 답안지 파일 (고객명, 점수, 주요 이유, 최종 결과)
출력 2: 원본 파일 + 점수/주요이유/최종결과 컬럼 추가
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────────
INPUT_PATH  = r"C:\Users\jhcho\Downloads\보험계리사\학회\삼성화재\AutoInsuranceClaims_수식오류수정.xlsx"
OUT_DIR     = r"C:\Users\jhcho\Downloads\보험계리사\학회\삼성화재"
OUT_ANSWER  = r"C:\Users\jhcho\Downloads\보험계리사\학회\삼성화재\인수심사_정답지.xlsx"
OUT_UPDATED = r"C:\Users\jhcho\Downloads\보험계리사\학회\삼성화재\raw data_scored.xlsx"

# ── 분위수 기준 (v2.2 전체 통합 기준, n=9134) ──────────────────
Q50 = 518.3    # 중위값
Q75 = 739.1    # 75%
Q90 = 1044.1   # 90%

# ── Rule 2 기준값 ──────────────────────────────────────────────
RULE2_CLAIM_THRESHOLD = Q75   # 75분위 이상
RULE3_CLAIM_THRESHOLD = Q90   # 90분위 이상


# ══════════════════════════════════════════════════════════════
# 섹션별 점수 함수
# ══════════════════════════════════════════════════════════════

def score_3_1_A(months_since_last_claim):
    """[3-1-A] 최근 청구 이력 (Months Since Last Claim) — 최대 25점"""
    m = months_since_last_claim
    if m >= 24:
        return 0,  "최근 청구 없음 (24개월↑)"
    elif m >= 12:
        return 5,  f"청구 {m}개월 전 (12~24개월)"
    elif m >= 6:
        return 15, f"청구 {m}개월 전 (6~12개월)"
    else:
        return 25, f"최근 청구 {m}개월 전 (6개월 미만)"


def score_3_1_B(total_claim):
    """[3-1-B] 총 청구금액 (Total Claim Amount) — 최대 35점"""
    if total_claim <= Q50:
        return 0,  f"청구금액 ${total_claim:.0f} (중위값 이하)"
    elif total_claim <= Q75:
        return 10, f"청구금액 ${total_claim:.0f} (50~75%)"
    elif total_claim <= Q90:
        return 20, f"청구금액 ${total_claim:.0f} (75~90%)"
    else:
        return 35, f"청구금액 ${total_claim:.0f} (상위 10%)"


def score_3_1_C(complaints):
    """[3-1-C] 미해결 민원 (Number of Open Complaints) — 최대 40점"""
    if complaints == 0:
        return 0,  "미해결 민원 없음"
    elif complaints == 1:
        return 10, "미해결 민원 1건"
    elif complaints == 2:
        return 25, "미해결 민원 2건"
    else:
        return 40, f"미해결 민원 {complaints}건 (3건↑)"


def score_3_2_A(vehicle_class):
    """[3-2-A] 차량 등급 (Vehicle Class) — 최대 25점"""
    vc = str(vehicle_class).strip()
    high_risk = {"Sports Car", "Luxury Car", "Luxury SUV"}
    if vc in high_risk:
        return 25, f"고위험 차종 ({vc})"
    elif vc == "SUV":
        return 10, "SUV (중간 위험)"
    elif vc in {"Four-Door Car", "Two-Door Car"}:
        return 0,  f"일반 승용차 ({vc})"
    else:
        return 10, f"차량등급 확인 필요 ({vc})"


def score_3_2_B(vehicle_size):
    """[3-2-B] 차량 크기 (Vehicle Size) — 최대 10점"""
    vs = str(vehicle_size).strip()
    if vs == "Large":
        return 10, "대형 차량"
    else:
        return 0,  f"차량 크기 {vs}"


def score_3_3_A(months_since_inception):
    """[3-3-A] 계약 경과 기간 (Months Since Policy Inception) — 최대 10점"""
    m = months_since_inception
    if m < 6:
        return 10, f"신규 계약 {m}개월 (6개월 미만)"
    elif m < 12:
        return 5,  f"계약 {m}개월 (6~12개월)"
    else:
        return 0,  f"계약 {m}개월 (12개월↑)"


def score_3_4_B(location):
    """[3-4-B] 거주 지역 유형 (Location) — 최대 5점"""
    if str(location).strip() == "Suburban":
        return 5, "Suburban 거주"
    else:
        return 0, f"지역 {location}"


def score_complex_rules(vehicle_class, months_last_claim, total_claim, complaints, months_inception):
    """복합 리스크 Rule 1~3"""
    rules_triggered = []
    bonus = 0

    # Rule 1: 고위험 차종 + 최근 청구 (12개월 미만)
    high_risk_vc = {"Sports Car", "Luxury Car", "Luxury SUV"}
    if str(vehicle_class).strip() in high_risk_vc and months_last_claim < 12:
        bonus += 20
        rules_triggered.append("Rule1(고위험차종+최근청구+20)")

    # Rule 2: 고액 청구 + 민원 2건↑
    if total_claim >= RULE2_CLAIM_THRESHOLD and complaints >= 2:
        bonus += 15
        rules_triggered.append("Rule2(고액청구+민원+15)")

    # Rule 3: 신규 계약 + 고액 청구
    if months_inception < 6 and total_claim >= RULE3_CLAIM_THRESHOLD:
        bonus += 10
        rules_triggered.append("Rule3(신규+고액청구+10)")

    return bonus, rules_triggered


# ══════════════════════════════════════════════════════════════
# 판정 등급
# ══════════════════════════════════════════════════════════════

def get_grade(total_score):
    if total_score <= 25:
        return "일반 인수"
    elif total_score <= 55:
        return "추가 심사"
    elif total_score <= 85:
        return "조건부 인수"
    else:
        return "인수 보류"


# ══════════════════════════════════════════════════════════════
# 행 단위 채점
# ══════════════════════════════════════════════════════════════

def score_row(row):
    reasons = []

    s_A, r_A = score_3_1_A(row["Months Since Last Claim"])
    s_B, r_B = score_3_1_B(row["Total Claim Amount"])
    s_C, r_C = score_3_1_C(row["Number of Open Complaints"])
    s_2A, r_2A = score_3_2_A(row["Vehicle Class"])
    s_2B, r_2B = score_3_2_B(row["Vehicle Size"])
    s_3A, r_3A = score_3_3_A(row["Months Since Policy Inception"])
    s_4B, r_4B = score_3_4_B(row["Location"])

    base = s_A + s_B + s_C + s_2A + s_2B + s_3A + s_4B

    bonus, rules = score_complex_rules(
        row["Vehicle Class"],
        row["Months Since Last Claim"],
        row["Total Claim Amount"],
        row["Number of Open Complaints"],
        row["Months Since Policy Inception"],
    )

    total = base + bonus

    # 주요 이유: 점수 높은 항목 top3 + 발동 Rule
    scored_items = sorted([
        (s_A, r_A), (s_B, r_B), (s_C, r_C),
        (s_2A, r_2A), (s_2B, r_2B), (s_3A, r_3A), (s_4B, r_4B),
    ], reverse=True)

    top_reasons = [r for s, r in scored_items if s > 0][:3]
    if rules:
        top_reasons += rules
    reason_str = " | ".join(top_reasons) if top_reasons else "위험 요인 없음"

    grade = get_grade(total)

    return pd.Series({
        "총점": total,
        "주요이유": reason_str,
        "최종결과": grade,
    })


# ══════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════

def main():
    print(f"▶ 파일 읽는 중: {INPUT_PATH}")
    df = pd.read_excel(INPUT_PATH)
    print(f"  → {len(df):,}건 로드 완료")

    print("▶ 채점 중...")
    scores = df.apply(score_row, axis=1)

    # ── 출력 1: 답안지 (고객명, 점수, 주요이유, 최종결과) ──────
    answer = pd.concat([df[["Customer"]], scores], axis=1)
    answer.columns = ["고객명", "점수", "주요이유", "최종결과"]
    answer.to_excel(OUT_ANSWER, index=False)
    print(f"  → 정답지 저장: {OUT_ANSWER}")

    # ── 출력 2: 원본 + 3개 컬럼 추가 ──────────────────────────
    df_out = pd.concat([df, scores], axis=1)
    df_out.to_excel(OUT_UPDATED, index=False)
    print(f"  → 원본+점수 저장: {OUT_UPDATED}")

    # ── 집계 요약 ──────────────────────────────────────────────
    print("\n══ 판정 분포 ══")
    dist = scores["최종결과"].value_counts()
    order = ["일반 인수", "추가 심사", "조건부 인수", "인수 보류"]
    for g in order:
        n = dist.get(g, 0)
        print(f"  {g:10s}: {n:5,}건 ({n/len(df)*100:.1f}%)")
    print(f"  {'합계':10s}: {len(df):5,}건")

    print("\n══ 점수 기술통계 ══")
    print(scores["총점"].describe().round(1).to_string())
    print("\n▶ 완료!")


if __name__ == "__main__":
    main()
