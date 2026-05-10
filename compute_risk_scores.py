"""
compute_risk_scores.py
======================

Conformal Prediction을 위한 Risk Score 계산 (실험 흐름의 3·4단계).

각 고객의 3회 inference 결과를 기반으로
  - Data Grounding   (DG) : 입력 데이터와 출력 입력데이터의 일치도
  - Policy Grounding (PG) : 룰 테이블 기준 정책 적용·점수 반영의 정확성
  - Stability             : 3회 반복 시 변수 조합·룰 조합의 일관성
을 계산하고, 최종 Risk Score를 산출.

  Risk_i = 1 - (w1·DG_i + w2·PG_i + w3·Stability_i)

특수 케이스 처리
----------------
주요근거가 비어있는 회차(AI가 위험점수=0으로 판정)에 대해서는 룰 테이블로
data.csv의 정답 점수를 직접 계산해 AI의 위험점수와 비교:
  - 일치(AI가 옳게 0점이라 판단)        → DG=PG=1.0
  - 불일치(AI가 점수를 누락한 hallucination) → DG=PG=0.0

가중치 w1, w2, w3은 코드 상단 WEIGHTS 상수로 자유롭게 조정할 수 있도록
설계 (calibration / test split 후 quantile 분석에 사용 예정).

선행 조건
---------
- 같은 폴더에 다음 파일이 있어야 함:
    data.csv                                     (입력 고객 데이터)
    underwriting_rules_scoring.csv               (룰 테이블)
    gemini_mesh_rag_experiment_results.json      (회차 1)
    gemini_mesh_rag_experiment2_results.json     (회차 2)
    gemini_mesh_rag_experiment3_results.json     (회차 3)
- 단, JSON 3개는 '판단기준' 값이 영어로 변환된 버전이어야 함 (룰 테이블의
  feature 컬럼과 일치). 한국어 판단기준이 섞여 있으면 매칭이 실패함.

사용법
------
    python compute_risk_scores.py

출력
----
    risk_scores.csv : Customer / DG / PG / Stability / RiskScore (5 컬럼)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# ============================================================================
# 설정
# ============================================================================
INPUT_DIR  = Path(".")          # 같은 폴더에서 실행 — 필요 시 절대경로로 변경
DATA_CSV   = INPUT_DIR / "data.csv"
RULES_CSV  = INPUT_DIR / "underwriting_rules_scoring.csv"
JSON_FILES = [
    INPUT_DIR / "rag_results1.json",
    INPUT_DIR / "rag_results2.json",
    INPUT_DIR / "rag_results3.json",
]
OUTPUT_CSV = INPUT_DIR / "rag_risk_scores.csv"

# Risk Score 가중치 — 추후 실험에서 자유롭게 조정 (합이 1일 필요는 없음)
WEIGHTS = {
    "w1_dg":        0.1,
    "w2_pg":        0.9,
    "w3_stability": 0.0,
}

# Stability 내부 가중치 (이미지 4: 0.5 / 0.5)
STABILITY_WEIGHTS = {
    "data_consistency": 0.5,
    "rule_consistency": 0.5,
}

# 수치 비교 허용오차 (Total Claim Amount 같은 float 잔여 흡수)
NUM_REL_TOL = 1e-3
NUM_ABS_TOL = 0.5


# ============================================================================
# 1. 로딩
# ============================================================================
def load_data():
    df_data  = pd.read_csv(DATA_CSV)
    df_rules = pd.read_csv(RULES_CSV)

    # 룰 테이블에서 점수 가산 룰만 남김 (final_judgment 등은 항목별 평가에서 제외)
    score_rules = df_rules[
        df_rules["action_type"].isin(["risk_add", "composite_risk_add"])
    ].copy()

    inferences = []
    for fp in JSON_FILES:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        # Customer ID로 빠른 lookup
        inferences.append({item["Customer"]: item for item in data["results"]})

    return df_data, score_rules, inferences


# ============================================================================
# 2. 값 비교 (Data Grounding 용)
# ============================================================================
def values_match(actual, claimed) -> bool:
    """data.csv 값과 LLM이 출력한 값을 허용오차 내에서 비교."""
    # NaN 처리
    if actual is None or (isinstance(actual, float) and math.isnan(actual)):
        return claimed is None or claimed == "" or (
            isinstance(claimed, float) and math.isnan(claimed)
        )
    if claimed is None:
        return False

    # 양쪽이 숫자로 캐스팅되면 허용오차 비교, 아니면 문자열 비교
    try:
        a = float(actual); c = float(claimed)
        return math.isclose(a, c, rel_tol=NUM_REL_TOL, abs_tol=NUM_ABS_TOL)
    except (TypeError, ValueError):
        return str(actual).strip() == str(claimed).strip()


# ============================================================================
# 3. 룰 condition 평가기
# ============================================================================
def eval_condition(cond_str: str, input_data: dict) -> bool:
    """CSV의 condition 문자열을 입력데이터 dict로 평가.

    지원:
      - 비교 연산자: >=, <=, >, <, ==, !=
      - 집합 멤버십: IN ('a', 'b', ...)
      - NULL/UNKNOWN: IS NULL OR UNKNOWN
      - AND 결합: 'A AND B AND C'
      - 복합 룰의 원자: feature 이름이 atom 앞에 명시 ('Vehicle Class IN (...)')
      - 단순 룰의 원자: feature 이름 생략 ('>= 24', input_data가 단일 키일 때)
    """
    if not isinstance(cond_str, str):
        return False
    parts = [p.strip() for p in re.split(r"\s+AND\s+", cond_str)]
    return all(_eval_atom(p, input_data) for p in parts)


def _eval_atom(atom: str, input_data: dict) -> bool:
    atom = atom.strip()
    feature = None
    rest = atom

    # atom 앞에 명시된 feature 이름이 있는지 확인 (가장 긴 매치 우선)
    if input_data:
        for k in sorted(input_data.keys(), key=len, reverse=True):
            if atom.startswith(k):
                feature = k
                rest = atom[len(k):].strip()
                break

    # 명시 없으면 단일 feature 입력일 때만 implicit 사용
    if feature is None:
        if len(input_data) == 1:
            feature = next(iter(input_data.keys()))
            rest = atom
        else:
            return False

    return _apply_op(input_data.get(feature), rest)


def _apply_op(val, expr: str) -> bool:
    expr = expr.strip()

    if "IS NULL" in expr.upper() or "UNKNOWN" in expr.upper():
        return val is None or val == "" or (
            isinstance(val, float) and math.isnan(val)
        )

    m = re.match(r"IN\s*\((.*)\)", expr, re.IGNORECASE)
    if m:
        items = [s.strip().strip("'\"") for s in m.group(1).split(",")]
        return str(val).strip() in items

    m = re.match(r"(>=|<=|==|!=|>|<)\s*(.+)", expr)
    if m:
        op, rhs_raw = m.group(1), m.group(2).strip()
        rhs = rhs_raw.strip("'\"")
        try:
            lhs_n = float(val); rhs_n = float(rhs)
            if op == ">=": return lhs_n >= rhs_n
            if op == "<=": return lhs_n <= rhs_n
            if op == ">":  return lhs_n >  rhs_n
            if op == "<":  return lhs_n <  rhs_n
            if op == "==": return math.isclose(lhs_n, rhs_n,
                                               rel_tol=NUM_REL_TOL, abs_tol=NUM_ABS_TOL)
            if op == "!=": return not math.isclose(lhs_n, rhs_n,
                                                   rel_tol=NUM_REL_TOL, abs_tol=NUM_ABS_TOL)
        except (TypeError, ValueError):
            sval = str(val).strip()
            if op == "==": return sval == rhs
            if op == "!=": return sval != rhs
            return False
    return False


# ============================================================================
# 4. rule_id 매핑 (PG 정답 판정 + Stability 룰 시그너처 공통 사용)
# ============================================================================
def find_rule_id(judgment_basis, input_data, reflected_score, df_rules):
    """LLM의 (판단기준, 입력데이터, 반영점수)로부터 CSV의 rule_id를 도출.

    절차
      1. feature == 판단기준 인 룰 후보 추출 (risk_add / composite_risk_add 만)
      2. action_value 가 반영점수와 (허용오차 내) 일치하는 후보로 좁힘
      3. 후보 중 condition이 입력데이터를 만족하는 첫 룰의 rule_id 반환
      4. 셋 다 통과하지 못하면 None
    """
    cands = df_rules[df_rules["feature"] == judgment_basis]
    if len(cands) == 0:
        return None

    score_matched = []
    for _, row in cands.iterrows():
        try:
            if math.isclose(float(row["action_value"]), float(reflected_score),
                            rel_tol=NUM_REL_TOL, abs_tol=NUM_ABS_TOL):
                score_matched.append(row)
        except (TypeError, ValueError):
            continue
    if not score_matched:
        return None

    for row in score_matched:
        if eval_condition(row["condition"], input_data or {}):
            return row["rule_id"]
    return None





# ============================================================================
# 5. 정답 점수 계산 (빈 주요근거 케이스의 PG/DG 검증용)
# ============================================================================
def compute_ground_truth_score(customer_row, df_rules) -> int:
    """data.csv 한 행에 룰 테이블을 적용해서 정답 위험점수 계산.

    - 단일 feature 룰(risk_add): feature별로 조건 만족하는 첫 룰의 action_value 가산
      (분위수·구간 기반이므로 feature당 정확히 한 룰만 매칭됨)
    - 복합 룰(composite_risk_add): 독립적으로 모두 평가, 만족 시 가산
    """
    total = 0

    # 단일 feature 룰
    single = df_rules[df_rules["action_type"] == "risk_add"]
    for feat in single["feature"].unique():
        feat_rules = single[single["feature"] == feat]
        in_data = {feat: customer_row.get(feat)}
        for _, rule in feat_rules.iterrows():
            if eval_condition(rule["condition"], in_data):
                try:
                    total += int(rule["action_value"])
                except (TypeError, ValueError):
                    pass
                break  # feature당 한 룰만

    # 복합 룰: 각 룰이 명시한 feature들을 input_data로 구성
    composite = df_rules[df_rules["action_type"] == "composite_risk_add"]
    for _, rule in composite.iterrows():
        feats = [f.strip() for f in str(rule["feature"]).split(",")]
        in_data = {f: customer_row.get(f) for f in feats}
        if eval_condition(rule["condition"], in_data):
            try:
                total += int(rule["action_value"])
            except (TypeError, ValueError):
                pass

    return total


# ============================================================================
# 6. 회차별 DG / PG 계산
# ============================================================================
def compute_dg_one(item, customer_row, df_rules):
    """한 회차의 Data Grounding.

    - 주요근거가 비어 있으면: 정답 점수를 계산해서 AI의 위험점수와 비교
      → 일치 시 DG=1.0, 불일치 시 DG=0.0
    - 그 외: 항목별 입력데이터 vs data.csv 일치 비율
    """
    if item is None:
        return 0.0
    items = item.get("주요근거") or []
    if not items:
        # 빈 주요근거: AI의 위험점수가 정답과 일치하는지 검증
        gt = compute_ground_truth_score(customer_row, df_rules)
        ai = item.get("위험점수")
        try:
            return 1.0 if int(ai) == int(gt) else 0.0
        except (TypeError, ValueError):
            return 0.0

    correct = total = 0
    for r in items:
        for k, claimed in (r.get("입력데이터") or {}).items():
            total += 1
            if values_match(customer_row.get(k), claimed):
                correct += 1
    return correct / total if total else 0.0


def compute_pg_one(item, customer_row, df_rules):
    """한 회차의 Policy Grounding (All-or-nothing).

    - 주요근거가 비어 있으면: 정답 점수와 AI 위험점수 일치 시 1.0, 불일치 시 0.0
      (DG와 동일 기준 — 빈 출력의 정당성을 정답 점수로만 판단)
    - 그 외: 각 항목에 대해 (판단기준, 입력데이터, 반영점수)로 rule_id 매핑
      성공하면 1, 실패하면 0. 매핑 성공은 다음 셋이 모두 충족됨을 의미:
        · 판단기준이 룰 테이블 feature에 존재
        · 입력데이터가 condition을 실제로 만족
        · 반영점수가 룰의 action_value와 일치
    """
    if item is None:
        return 0.0
    items = item.get("주요근거") or []
    if not items:
        gt = compute_ground_truth_score(customer_row, df_rules)
        ai = item.get("위험점수")
        try:
            return 1.0 if int(ai) == int(gt) else 0.0
        except (TypeError, ValueError):
            return 0.0

    correct = total = 0
    for r in items:
        total += 1
        feature = r.get("판단기준")
        score   = (r.get("적용정책") or {}).get("반영점수")
        in_data = r.get("입력데이터") or {}
        if find_rule_id(feature, in_data, score, df_rules) is not None:
            correct += 1
    return correct / total if total else 0.0


def extract_signatures(item, df_rules):
    """한 회차에서 (변수 조합 frozenset, 룰 조합 frozenset)을 추출."""
    if item is None:
        return frozenset(), frozenset()
    var_set, rule_set = set(), set()
    for r in item.get("주요근거", []):
        for k in (r.get("입력데이터") or {}).keys():
            var_set.add(k)
        feature = r.get("판단기준")
        score   = (r.get("적용정책") or {}).get("반영점수")
        in_data = r.get("입력데이터") or {}
        rid = find_rule_id(feature, in_data, score, df_rules)
        # 매핑 실패는 'UNGROUNDED' 시그너처로 별도 추적
        # → 3회 모두 동일한 잘못된 출력이면 stability에는 반영 (consistent하게 틀림)
        if rid is None:
            rid = f"UNGROUNDED::{feature}::{score}"
        rule_set.add(rid)
    return frozenset(var_set), frozenset(rule_set)


# ============================================================================
# 7. 고객 단위 집계 (DG, PG 평균 + Stability + Risk)
# ============================================================================
def compute_per_customer(customer, customer_row, items_3runs, df_rules):
    n = len(items_3runs)
    dg_runs, pg_runs, var_sigs, rule_sigs = [], [], [], []

    for item in items_3runs:
        dg_runs.append(compute_dg_one(item, customer_row, df_rules))
        pg_runs.append(compute_pg_one(item, customer_row, df_rules))
        v, r = extract_signatures(item, df_rules)
        var_sigs.append(v)
        rule_sigs.append(r)

    dg_avg = sum(dg_runs) / n
    pg_avg = sum(pg_runs) / n

    # Stability: 가장 많이 등장한 시그너처의 빈도 / 전체 회차 수
    data_consistency = max(Counter(var_sigs).values())  / n
    rule_consistency = max(Counter(rule_sigs).values()) / n
    stability = (
        STABILITY_WEIGHTS["data_consistency"] * data_consistency
        + STABILITY_WEIGHTS["rule_consistency"] * rule_consistency
    )

    risk = 1.0 - (
        WEIGHTS["w1_dg"]        * dg_avg
        + WEIGHTS["w2_pg"]        * pg_avg
        + WEIGHTS["w3_stability"] * stability
    )

    return {
        "Customer":  customer,
        "DG":        dg_avg,
        "PG":        pg_avg,
        "Stability": stability,
        "RiskScore": risk,
    }


# ============================================================================
# 8. 메인
# ============================================================================
def main():
    print("=" * 72)
    print("Conformal Prediction — Risk Score 계산 (실험 흐름 3·4단계)")
    print("=" * 72)

    # 7-1. 로딩
    df_data, df_rules, inferences = load_data()
    print(f"\n• 입력 고객 데이터 (data.csv)        : {len(df_data):>4} 행")
    print(f"• 룰 테이블 (점수 가산 룰만 사용) : {len(df_rules):>4} 룰")
    for i, inf in enumerate(inferences, 1):
        print(f"• Inference {i} ({JSON_FILES[i-1].name})  : {len(inf):>4} 고객")

    # 7-2. 고객 ID 정합성 점검
    customers_data = set(df_data["Customer"])
    customers_inf  = [set(inf.keys()) for inf in inferences]
    common = customers_data
    for cs in customers_inf:
        common &= cs
    skipped = customers_data - common
    if skipped:
        print(f"\n⚠ {len(skipped)}명의 고객이 일부 inference에 누락 — 결과에서 제외")
        for c in sorted(skipped)[:5]:
            in_files = [i + 1 for i, cs in enumerate(customers_inf) if c in cs]
            print(f"   {c} (등장 회차: {in_files})")

    print(f"\n• 평가 대상 (3회 모두 등장)         : {len(common):>4} 명")

    # 7-3. 고객별 계산 루프
    df_data_idx = df_data.set_index("Customer")
    results = []
    # 빈 주요근거 통계 추적
    empty_correct_runs = 0
    empty_wrong_runs   = 0
    for cust in sorted(common):
        customer_row = df_data_idx.loc[cust].to_dict()
        items_3runs  = [inf.get(cust) for inf in inferences]
        # 빈 주요근거 회차 추적
        gt = compute_ground_truth_score(customer_row, df_rules)
        for it in items_3runs:
            if it is not None and not it.get("주요근거"):
                try:
                    if int(it.get("위험점수")) == int(gt):
                        empty_correct_runs += 1
                    else:
                        empty_wrong_runs += 1
                except (TypeError, ValueError):
                    empty_wrong_runs += 1
        results.append(
            compute_per_customer(cust, customer_row, items_3runs, df_rules)
        )

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✓ {len(df_out)} 고객의 risk score → {OUTPUT_CSV}")

    # 빈 주요근거 케이스 검증 통계
    total_empty = empty_correct_runs + empty_wrong_runs
    if total_empty > 0:
        print(f"\n빈 주요근거 회차 검증 (정답 점수 vs AI 위험점수):")
        print(f"  • 일치 (DG=PG=1.0 처리): {empty_correct_runs:>4} 회차")
        print(f"  • 불일치 (DG=PG=0.0 처리): {empty_wrong_runs:>4} 회차  ← 점수 누락 의심")
        print(f"  • 합계: {total_empty} 회차")

    # 7-4. 요약 통계
    print("\n" + "=" * 72)
    print("요약 통계")
    print("=" * 72)
    for col in ["DG", "PG", "Stability", "RiskScore"]:
        s = df_out[col]
        print(f"  {col:<10} mean={s.mean():.4f}  std={s.std():.4f}  "
              f"min={s.min():.4f}  P50={s.median():.4f}  "
              f"P90={s.quantile(0.90):.4f}  max={s.max():.4f}")

    print(f"\n현재 가중치: w1(DG)={WEIGHTS['w1_dg']:.4f}  "
          f"w2(PG)={WEIGHTS['w2_pg']:.4f}  "
          f"w3(Stab)={WEIGHTS['w3_stability']:.4f}")
    print("→ 가중치 변경: 코드 상단 WEIGHTS dict 수정 후 재실행")


if __name__ == "__main__":
    main()
