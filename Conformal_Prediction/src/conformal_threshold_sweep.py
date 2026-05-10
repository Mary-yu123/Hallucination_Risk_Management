"""
conformal_threshold_sweep.py
============================

Conformal Prediction의 5~7단계 실행:
- 5단계: 데이터를 calibration 50% / test 50%로 split
- 6단계: cal data로 risk score의 quantile 계산해서 threshold 결정
- 7단계: 다양한 coverage 수준 (0.1, 0.2, ..., 0.9)에서 test data를 분류
        threshold 이하 → AI 자동 처리
        threshold 초과 → 사람 언더라이터에게 이관

Coverage 정의 (Conformal Prediction)
------------------------------------
- coverage α ∈ (0, 1) = AI 자동 처리 목표 비율
- threshold τ = cal risk scores의 α-quantile
- 직관:
    coverage 0.1 → threshold 매우 낮음 → 거의 다 사람에게 (가장 엄격)
    coverage 0.9 → threshold 높음 → 대부분 AI 자동 (가장 느슨)

선행 조건
---------
- compute_risk_scores.py 실행 후 생성된 risk_scores.csv가 있어야 함
- error rate 계산을 위해 원본 data.csv, rules CSV, 3개 inference JSON 필요

출력
----
threshold_sweep.csv :
  coverage, threshold,
  n_test, n_ai_auto, pct_ai_auto, n_human_review, pct_human_review,
  error_rate_in_ai_auto, error_rate_in_human_review

→ 꺾은 선 그래프: x=threshold (또는 coverage),
   y1=pct_human_review, y2=error_rate_in_ai_auto
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from compute_risk_scores import (
    INPUT_DIR,
    load_data,
    compute_dg_one,
    compute_pg_one,
    compute_ground_truth_score,
)

# ============================================================================
# 설정
# ============================================================================
RISK_SCORES_CSV = INPUT_DIR / "rag_risk_scores.csv"
OUT_CSV         = INPUT_DIR / "rag_threshold_sweep.csv"

CAL_RATIO   = 0.5
SPLIT_SEED  = 42
COVERAGE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ============================================================================
# 고객별 binary error label 계산
# ============================================================================
def compute_error_labels(df_data, df_rules, inferences, customers):
    """3회 inference 중 1회라도 error가 있으면 1, 모두 정상이면 0.

    Error 정의 (회차당, 첨부 이미지 명세 그대로):
      - 입력 데이터 왜곡        → DG < 1.0
      - 존재하지 않는 근거 생성  → PG에 포함 (룰 매칭 실패)
      - 정책 점수 오적용        → PG에 포함 (action_value 불일치)
      - underwriting guideline 위반 → AI 위험점수 != 정답 점수
    """
    df_data_idx = df_data.set_index("Customer")
    labels = {}
    for cust in customers:
        if cust not in df_data_idx.index:
            labels[cust] = None
            continue
        customer_row = df_data_idx.loc[cust].to_dict()
        items = [inf.get(cust) for inf in inferences]
        gt = compute_ground_truth_score(customer_row, df_rules)

        any_error = False
        for item in items:
            dg = compute_dg_one(item, customer_row, df_rules)
            pg = compute_pg_one(item, customer_row, df_rules)
            err = (dg < 0.999) or (pg < 0.999)
            if not err:
                ai_score = (item or {}).get("위험점수")
                try:
                    if int(ai_score) != int(gt):
                        err = True
                except (TypeError, ValueError):
                    err = True
            if err:
                any_error = True
                break  # 1회만 error 있어도 충분
        labels[cust] = int(any_error)
    return labels


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 72)
    print("Conformal Prediction — 5~7단계 (Threshold Sweep)")
    print("=" * 72)

    # ── 1. risk_scores.csv 로드 ─────────────────────────────────────
    df_risk = pd.read_csv(RISK_SCORES_CSV)
    df_risk.columns = df_risk.columns.str.strip()  # BOM 처리
    print(f"\n• risk_scores.csv 로드 : {len(df_risk)} 고객")
    print(f"  RiskScore  mean={df_risk['RiskScore'].mean():.4f}  "
          f"min={df_risk['RiskScore'].min():.4f}  "
          f"max={df_risk['RiskScore'].max():.4f}")

    # ── 2. binary error label 계산 ─────────────────────────────────
    print("\n• 고객별 binary error label 계산 중... (수십 초)")
    df_data, df_rules, inferences = load_data()
    labels = compute_error_labels(
        df_data, df_rules, inferences, df_risk["Customer"].tolist()
    )
    df_risk["error"] = df_risk["Customer"].map(labels)
    df_risk = df_risk.dropna(subset=["error"]).copy()
    df_risk["error"] = df_risk["error"].astype(int)

    n_err = int(df_risk["error"].sum())
    print(f"  Error=1 (1회 이상 hallucination): "
          f"{n_err} / {len(df_risk)} ({n_err/len(df_risk)*100:.1f}%)")

    # ── 3. 50/50 split (5단계) ─────────────────────────────────────
    shuffled = df_risk.sample(frac=1, random_state=SPLIT_SEED).reset_index(drop=True)
    n_cal = int(len(shuffled) * CAL_RATIO)
    cal  = shuffled.iloc[:n_cal].reset_index(drop=True)
    test = shuffled.iloc[n_cal:].reset_index(drop=True)

    print(f"\n• 5단계 — Split (seed={SPLIT_SEED}, ratio={CAL_RATIO})")
    print(f"    cal  : {len(cal):>3} 명  (error rate {cal['error'].mean()*100:.1f}%)")
    print(f"    test : {len(test):>3} 명  (error rate {test['error'].mean()*100:.1f}%)")

    # ── 4. Coverage sweep (6~7단계) ───────────────────────────────
    print(f"\n• 6~7단계 — Coverage 수준별 threshold sweep")

    rows = []
    for cov in COVERAGE_LEVELS:
        # cal data의 cov-quantile = threshold
        threshold = float(np.quantile(cal["RiskScore"], cov, method="linear"))

        # test data 분류
        ai_auto = test[test["RiskScore"] < threshold]
        human   = test[test["RiskScore"] >=  threshold]

        rows.append({
            "coverage":                cov,
            "threshold":               threshold,
            "n_test":                  len(test),
            "n_ai_auto":               len(ai_auto),
            "pct_ai_auto":             len(ai_auto) / len(test),
            "n_human_review":          len(human),
            "pct_human_review":        len(human) / len(test),
            "error_rate_in_ai_auto":   (ai_auto["error"].mean()
                                        if len(ai_auto) else float("nan")),
            "error_rate_in_human_review": (human["error"].mean()
                                           if len(human) else float("nan")),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✓ {OUT_CSV}")

    # ── 5. 화면 요약 ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("결과 요약")
    print("=" * 72)
    print(
        f"{'coverage':>9} {'threshold':>10} "
        f"{'AI auto':>8} {'human':>8} "
        f"{'err in AI':>10} {'err in human':>13}"
    )
    print("-" * 72)
    for _, r in df_out.iterrows():
        err_ai = (f"{r['error_rate_in_ai_auto']*100:>5.1f}%"
                  if pd.notna(r["error_rate_in_ai_auto"]) else "  n/a ")
        err_hu = (f"{r['error_rate_in_human_review']*100:>5.1f}%"
                  if pd.notna(r["error_rate_in_human_review"]) else "  n/a ")
        print(
            f"{r['coverage']:>9.1f} "
            f"{r['threshold']:>10.4f} "
            f"{r['pct_ai_auto']*100:>7.1f}% "
            f"{r['pct_human_review']*100:>7.1f}% "
            f"     {err_ai} "
            f"      {err_hu}"
        )

    print("\n해석:")
    print("  • coverage ↑ → threshold ↑ → AI 자동 처리 ↑ → 사람 검토 ↓ → AI에서 새는 오류 ↑")
    print("  • '엄격한 보험사' = 낮은 coverage (0.1~0.3)")
    print("  • '자동화 우선 보험사' = 높은 coverage (0.7~0.9)")
    print("\n발표용 꺾은선 그래프:")
    print(f"  파일: {OUT_CSV.name}")
    print(f"  x축: threshold (또는 coverage)")
    print(f"  y축 1: pct_human_review (사람 검토 비율)")
    print(f"  y축 2: error_rate_in_ai_auto (AI 자동 영역의 실제 오류율)")


if __name__ == "__main__":
    main()
