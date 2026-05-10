"""
evaluate_weight_combos.py
=========================

가중치 조합 (w1, w2, w3) 별로 Risk Score를 계산하고, 실제 hallucination/오류율과의
상관성을 비교하여 최적 조합을 탐색.

Error 정의 (회차당, 첨부 이미지 명세 그대로)
--------------------------------------------
다음 중 하나라도 발생하면 Error = 1:
  - 입력 데이터 왜곡          → DG < 1.0
  - 존재하지 않는 근거 생성    → PG에서 룰 매칭 실패 (PG에 포함됨)
  - 정책 점수 오적용          → 반영점수 != action_value (PG에 포함됨)
  - underwriting guideline 위반 → AI의 위험점수 != 룰북 정답 점수
모두 통과하면 Error = 0.

고객 단위 error_rate = (Error=1인 회차 수) / 3 ∈ {0, 1/3, 2/3, 1}

분석 절차
---------
1. 모든 고객에 대해 (DG_avg, PG_avg, Stability, error_rate) 사전 계산 — 가중치 무관
2. 가중치 grid (sum=1, step=0.1, 총 66개 조합)에 대해 Risk Score 산출
3. 각 조합에 대해 두 가지 평가:
   - Customer-level Spearman correlation (risk_score ↔ error_rate)
   - Risk bin (0-0.2, ..., 0.8-1.0) 별 평균 error rate (단조성)
4. Spearman corr 기준으로 최적 조합 선정

출력
----
- bin_error_rates.csv : Risk Bin × Error Rate (Top 5 조합, long format, 꺾은선 플롯용)
- weights_ranking.csv : 모든 가중치 조합의 메트릭 랭킹

선행 조건
---------
같은 폴더에 compute_risk_scores.py, data.csv, underwriting_rules_scoring.csv,
3개 inference JSON 파일이 있어야 함.
"""

from __future__ import annotations

import time
from collections import Counter

import pandas as pd

from compute_risk_scores import (
    INPUT_DIR,
    load_data,
    compute_dg_one,
    compute_pg_one,
    extract_signatures,
    compute_ground_truth_score,
    STABILITY_WEIGHTS,
)

# ============================================================================
# 설정
# ============================================================================
OUT_BINS    = INPUT_DIR / "rag_bin_error_rates.csv"
OUT_RANKING = INPUT_DIR / "rag_weights_ranking.csv"

# 첨부 이미지 그대로의 5개 bin
BIN_EDGES  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

WEIGHT_STEP = 0.1   # 0.1 → 66 조합, 0.05 → 231 조합
TOP_N       = 5     # 비교 출력에 포함할 상위 조합 수


# ============================================================================
# 가중치 grid 생성: w1 + w2 + w3 = 1, step 단위
# ============================================================================
def generate_weight_combos(step: float = WEIGHT_STEP):
    n = round(1 / step)
    combos = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            combos.append((round(i * step, 2),
                           round(j * step, 2),
                           round(k * step, 2)))
    return combos


# ============================================================================
# 고객 단위 사전 계산 — 가중치와 무관한 부분은 한 번만
# ============================================================================
def precompute_components(df_data, df_rules, inferences, customers):
    df_data_idx = df_data.set_index("Customer")
    rows = []
    for cust in sorted(customers):
        customer_row = df_data_idx.loc[cust].to_dict()
        items = [inf.get(cust) for inf in inferences]
        gt = compute_ground_truth_score(customer_row, df_rules)

        dg_runs, pg_runs, errors = [], [], []
        var_sigs, rule_sigs = [], []
        for item in items:
            dg = compute_dg_one(item, customer_row, df_rules)
            pg = compute_pg_one(item, customer_row, df_rules)

            # ─ 회차별 binary error 판정 (이미지 정의 그대로) ────────────
            err = (dg < 0.999) or (pg < 0.999)
            if not err:
                # underwriting guideline 위반: AI 위험점수 != 정답
                ai_score = (item or {}).get("위험점수")
                try:
                    if int(ai_score) != int(gt):
                        err = True
                except (TypeError, ValueError):
                    err = True

            dg_runs.append(dg)
            pg_runs.append(pg)
            errors.append(int(err))
            v, r = extract_signatures(item, df_rules)
            var_sigs.append(v)
            rule_sigs.append(r)

        n = len(items)
        dg_avg = sum(dg_runs) / n
        pg_avg = sum(pg_runs) / n
        dc = max(Counter(var_sigs).values())  / n
        rc = max(Counter(rule_sigs).values()) / n
        stability = (
            STABILITY_WEIGHTS["data_consistency"] * dc
            + STABILITY_WEIGHTS["rule_consistency"] * rc
        )
        rows.append({
            "Customer":   cust,
            "DG_avg":     dg_avg,
            "PG_avg":     pg_avg,
            "Stability":  stability,
            "error_rate": sum(errors) / n,
        })
    return pd.DataFrame(rows)


# ============================================================================
# 메트릭
# ============================================================================
def compute_auc(scores, binary_labels):
    """수동 AUC (rank-based, sklearn 의존성 제거)."""
    s = pd.Series(scores).reset_index(drop=True)
    y = pd.Series(binary_labels).reset_index(drop=True)
    if y.nunique() < 2:
        return float("nan")
    ranks = s.rank()
    pos = (y == 1)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compute_metrics(risk, error_rate):
    s = pd.Series(risk).reset_index(drop=True)
    e = pd.Series(error_rate).reset_index(drop=True)
    return {
        "spearman": s.corr(e, method="spearman"),
        "pearson":  s.corr(e, method="pearson"),
        "auc":      compute_auc(s, (e >= 0.5).astype(int)),
    }


def bin_summary(risk, error_rate, bin_edges, bin_labels):
    df = pd.DataFrame({"risk": pd.Series(risk).values,
                       "err":  pd.Series(error_rate).values})
    df["bin"] = pd.cut(df["risk"], bins=bin_edges,
                       include_lowest=True, right=True, labels=bin_labels)
    g = df.groupby("bin", observed=False)["err"]
    return g.mean(), g.count()


def bin_monotonicity(bin_means):
    valid = bin_means.dropna()
    if len(valid) < 2:
        return float("nan")
    idx = pd.Series(range(len(valid)))
    val = valid.reset_index(drop=True)
    return idx.corr(val, method="pearson")


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 72)
    print("Risk Score 가중치 조합 탐색 + Actual Error Rate 검증")
    print("=" * 72)

    df_data, df_rules, inferences = load_data()
    customers = set(df_data["Customer"])
    for inf in inferences:
        customers &= set(inf.keys())
    print(f"\n• 평가 대상 고객: {len(customers)} 명")

    print("• 회차별 DG/PG/Stability/Error 사전 계산 중... (수십 초 소요)")
    df_comp = precompute_components(df_data, df_rules, inferences, customers)

    err_dist = df_comp["error_rate"].value_counts().sort_index()
    print("\n고객 단위 error_rate 분포:")
    for r, cnt in err_dist.items():
        pct = cnt / len(df_comp) * 100
        print(f"  {r:.3f} ({int(round(r*3))}/3 회 오류): {cnt:>4} 명 ({pct:.1f}%)")
    n_bin1 = (df_comp["error_rate"] >= 0.5).sum()
    print(f"  → error_rate ≥ 0.5 (오류 다수, AUC 양성): {n_bin1} / {len(df_comp)}")

    combos = generate_weight_combos(WEIGHT_STEP)
    print(f"\n• 가중치 grid: {len(combos)}개 조합 (step={WEIGHT_STEP})")
    print("• 각 조합에 대해 Risk Score 계산 + 메트릭 평가 중...")

    rows = []
    for w1, w2, w3 in combos:
        risk = 1.0 - (w1 * df_comp["DG_avg"]
                      + w2 * df_comp["PG_avg"]
                      + w3 * df_comp["Stability"])
        m = compute_metrics(risk, df_comp["error_rate"])
        bin_means, bin_counts = bin_summary(
            risk, df_comp["error_rate"], BIN_EDGES, BIN_LABELS
        )
        mono = bin_monotonicity(bin_means)
        row = {
            "w1_dg": w1, "w2_pg": w2, "w3_stab": w3,
            "spearman": m["spearman"],
            "pearson":  m["pearson"],
            "auc":      m["auc"],
            "bin_monotonicity": mono,
        }
        for label in BIN_LABELS:
            row[f"bin_{label}"] = bin_means.get(label, float("nan"))
            row[f"n_{label}"]   = int(bin_counts.get(label, 0))
        rows.append(row)

    # Spearman 기준 랭킹 (커스토머 단위 단조성)
    df_rank = (pd.DataFrame(rows)
                 .sort_values("spearman", ascending=False)
                 .reset_index(drop=True))
    df_rank.insert(0, "rank", range(1, len(df_rank) + 1))
    df_rank.to_csv(OUT_RANKING, index=False, encoding="utf-8-sig")
    print(f"\n✓ 모든 조합 ranking → {OUT_RANKING}")

    # ─── 최고 조합 출력 ─────────────────────────────────────────────
    best = df_rank.iloc[0]
    print("\n" + "=" * 72)
    print("★ 최고 가중치 조합 (Spearman correlation 기준)")
    print("=" * 72)
    print(f"  w1 (DG)        = {best['w1_dg']:.2f}")
    print(f"  w2 (PG)        = {best['w2_pg']:.2f}")
    print(f"  w3 (Stability) = {best['w3_stab']:.2f}")
    print(f"\n  Spearman correlation : {best['spearman']:+.4f}")
    print(f"  Pearson correlation  : {best['pearson']:+.4f}")
    print(f"  AUC                  : {best['auc']:.4f}")
    print(f"  Bin 단조성 (Pearson) : {best['bin_monotonicity']:+.4f}")
    print(f"\n  Bin별 actual error rate:")
    for label in BIN_LABELS:
        rate = best[f"bin_{label}"]
        n    = best[f"n_{label}"]
        bar  = "█" * int(round(rate * 30)) if pd.notna(rate) else ""
        rs   = f"{rate:.4f}" if pd.notna(rate) else "(empty)"
        print(f"    {label}: {rs}  (n={n:>3})  {bar}")

    # ─── Top N 조합 비교 ───────────────────────────────────────────
    print(f"\nTop {TOP_N} 조합 (Spearman 순):")
    print(f"{'rank':>4}  {'w1':>5} {'w2':>5} {'w3':>5}  "
          f"{'spearman':>9}  {'pearson':>9}  {'AUC':>6}  {'monoton':>8}")
    for i in range(min(TOP_N, len(df_rank))):
        r = df_rank.iloc[i]
        print(f"{int(r['rank']):>4}  "
              f"{r['w1_dg']:>5.2f} {r['w2_pg']:>5.2f} {r['w3_stab']:>5.2f}  "
              f"{r['spearman']:>+9.4f}  {r['pearson']:>+9.4f}  "
              f"{r['auc']:>6.4f}  {r['bin_monotonicity']:>+8.4f}")

    # ─── 꺾은선 플롯용 long-format CSV ─────────────────────────────
    plot_rows = []
    for i in range(min(TOP_N, len(df_rank))):
        r = df_rank.iloc[i]
        combo_label = (f"rank{int(r['rank'])}_"
                       f"w{r['w1_dg']:.1f}_{r['w2_pg']:.1f}_{r['w3_stab']:.1f}")
        for j, label in enumerate(BIN_LABELS):
            plot_rows.append({
                "combo_label":  combo_label,
                "rank":         int(r["rank"]),
                "w1_dg":        r["w1_dg"],
                "w2_pg":        r["w2_pg"],
                "w3_stab":      r["w3_stab"],
                "risk_bin":     label,
                "bin_midpoint": (BIN_EDGES[j] + BIN_EDGES[j + 1]) / 2,
                "n_customers":  int(r[f"n_{label}"]),
                "error_rate":   r[f"bin_{label}"],
            })
    df_plot = pd.DataFrame(plot_rows)
    df_plot.to_csv(OUT_BINS, index=False, encoding="utf-8-sig")
    print(f"\n✓ Bin error rate (Top {TOP_N}, long format) → {OUT_BINS}")

    print(f"\n총 소요 시간: {time.time() - t0:.1f}초")
    print("\n" + "=" * 72)
    print("플롯 가이드")
    print("=" * 72)
    print(f"  파일 1: {OUT_BINS.name}")
    print(f"    x = bin_midpoint (또는 risk_bin 카테고리)")
    print(f"    y = error_rate")
    print(f"    line/color = combo_label  (rank=1 만 필터하면 단일 선)")
    print(f"\n  파일 2: {OUT_RANKING.name}")
    print(f"    전체 {len(df_rank)}개 조합의 메트릭 — 분석/검증용")


if __name__ == "__main__":
    main()
