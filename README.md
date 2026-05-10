# Hallucination Risk Management for Insurance Underwriting AI

LLM 기반 보험 인수심사에서 발생하는 **할루시네이션 리스크**를 정량화하고, 위험한 케이스만 사람에게 라우팅하는 시스템입니다.
**RAG + Data Mesh**로 입력 품질을 확보하고, **Conformal Prediction**으로 출력 신뢰도를 통계적으로 보장합니다.

## Background

LLM을 보험 인수심사(underwriting)에 도입할 때 가장 큰 장벽은 **할루시네이션**입니다. 보험 도메인은 ① 잘못된 결정이 손해율에 직결되는 고위험 영역이고, ② 명시적인 인수 규정(rule)이 존재하지만 LLM이 이를 일관되게 따르지 않으며, ③ 인수심사 기준서 기반으로 정답 점수가 계산 가능해 정량 평가가 가능합니다. 본 프로젝트는 LLM 출력을 무조건 신뢰하는 대신 **위험도를 측정하고 임계값을 통계적으로 정해** 안전한 자동화를 달성하는 것을 목표로 합니다.

## Methodology

### 1) Input Quality — RAG + Data Mesh

LLM 환각의 1차 원인은 **입력 데이터/룰의 모호성**입니다. 이를 두 축으로 차단합니다.

- **Data Mesh**: 데이터를 고객 정보 / 계약 / 청구·민원 / 차량 / CRM 도메인으로 분리·정제하여 신뢰도 높은 구조화 입력 제공
- **RAG**: 인수 규정 룰북에서 현재 케이스에 관련된 룰만 검색하여 컨텍스트로 주입

> RAG only vs RAG + Data Mesh 비교 결과, **RAG + Data Mesh에서 DG weight가 0일 때 ranking 성능이 최고**였습니다. 이는 별도 보정 없이도 LLM이 입력 데이터를 정확히 따라간다는, 즉 **입력 품질이 충분히 정제되어 있다**는 신호입니다.

### 2) Output Diagnosis — Risk Score

LLM 출력의 위험도를 직교하는 3개 메트릭으로 측정합니다.

- **DG (Data Grounding)** — LLM이 입력 고객 데이터에 충실한가 
- **PG (Policy Grounding)** — LLM이 인수 규정을 준수하는가 
- **Stability** — 동일 입력에 일관된 답변을 내는가

$$
\text{Risk}(x) = 1 - \big( w_1 \cdot \text{DG}(x) + w_2 \cdot \text{PG}(x) + w_3 \cdot \text{Stability}(x) \big)
$$

가중치 $(w_1, w_2, w_3)$는 grid search로 Spearman / AUC / bin monotonicity 기준 최적화합니다.

### 3) Safety Guarantee — Conformal Prediction

Risk Score만으로는 임계값이 임의적입니다. **Conformal Prediction**으로 통계적 보장 하에 임계값 $\tau$를 결정합니다.

1. Calibration set에서 Risk Score 분포를 학습
2. $(1-\alpha)$ quantile을 임계값 $\tau$로 설정
3. Test set에서 $\text{Risk} > \tau$ 인 케이스만 사람 검토로 라우팅

이 절차는 **진짜로 위험한 케이스의 최소 $(1-\alpha) \times 100\%$ 가 사람 검토로 라우팅된다**는 marginal coverage를 보장합니다.

## Pipeline

1. **Data Mesh 정제** — 원본 고객 데이터를 4개 도메인으로 구조화
2. **RAG 검색** — 인수 규정서에서 케이스별 관련 심사 기준 추출
3. **LLM 추론** — Gemini를 통해 동일 입력으로 3회 점수 예측
4. **Risk Score 계산** — DG / PG / Stability 가중합
5. **Conformal Prediction** — Calibration quantile로 사람 검토 분기

## Data

[**Auto Insurance Claims Updated to 2024**](https://www.kaggle.com/datasets/thebumpkin/auto-insurance-claims-updated-to-2024) (Kaggle)

- 9,134건의 자동차 보험 청구 데이터
- 34개 컬럼 (고객 인구통계, 차량 정보, 청구 이력, 보험료 등)
- 인수심사에 필요한 충분한 feature 다양성과 룰 기반 점수화 가능성을 모두 갖추어 선정

별도로 자체 제작한 **인수 규정 룰북**을 함께 활용하여 정답 점수와 PG(Policy Grounding) 평가에 사용합니다.

## Model

- **LLM**: Gemini (3회 반복 추론으로 Stability 측정)
- **RAG**: 인수 규정 룰북 기반 검색

## References

1. Angelopoulos, A. N., & Bates, S. (2023). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.*
2. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
3. Dehghani, Z. (2022). *Data Mesh: Delivering Data-Driven Value at Scale.* O'Reilly.
