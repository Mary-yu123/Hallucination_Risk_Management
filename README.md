# Hallucination Risk Management for Insurance Underwriting AI

A system for quantifying **hallucination risk** in LLM-based insurance underwriting and routing only high-risk cases to human reviewers. The framework improves input reliability through **RAG** and **Data Mesh** architecture, while statistically guaranteeing output confidence using **Conformal Prediction**.


## Background

Many industries are actively adopting AI, and the insurance sector is no exception. AI is already proving its value in various areas, including claims settlement, underwriting, product development, and customer service chatbots.
However, real-world implementation faces a critical risk: AI "hallucination," where the model generates false information and presents it as fact. Therefore, this study proposes a methodology to minimize hallucinations and enhance the reliability of AI underwriting models, an area that requires highly precise judgment.


## Methodology

### 1) Input Quality — RAG + Data Mesh

A primary cause of LLM hallucination is ambiguity in input data and business rules. We mitigate this through two complementary components:

- **Data Mesh**: Reducing hallucinations requires redesigning the data architecture so the AI relies only on trustworthy, structured data. A Data Mesh organizes underwriting data into domain-specific datasets, helping the AI better understand relationships between underwriting factors while improving reliability and consistency.
  
- **RAG(Retrieval-Augmented Generation)**: RAG forces the AI to generate responses only from predefined underwriting guidelines and retrieved reference documents, reducing unsupported assumptions and improving factual accuracy. We evaluated this approach using hypothetical underwriting guidelines and 500 records from a U.S. auto insurance claims dataset.
  
### 2) Output Diagnosis — Risk Score

LLM output risk is evaluated using three metrics:

- **DG (Data Grounding)** — Does the LLM faithfully follow the input customer data?
- **PG (Policy Grounding)** — Does the LLM comply with underwriting policies and guidelines?
- **Stability** — Does the LLM produce consistent responses for the same input?


$$
\text{Risk}(x) = 1 - \big( w_1 \cdot \text{DG}(x) + w_2 \cdot \text{PG}(x) + w_3 \cdot \text{Stability}(x) \big)
$$

The weights $(w_1, w_2, w_3)$ are optimized through grid search using Spearman correlation, AUC, and bin monotonicity as evaluation criteria.

### 3) Safety Guarantee — Conformal Prediction

Risk Scores alone require an arbitrary threshold selection. To address this, we use Conformal Prediction to determine the threshold $\tau$ with statistical guarantees.

1. Learn the Risk Score distribution from a calibration set
2. Set the $(1-\alpha)$ quantile as the threshold $\tau$
3. Route only cases with $\text{Risk} > \tau$ to human review
   
This procedure guarantees marginal coverage, ensuring that at least $(1-\alpha)\times100%$ of truly high-risk cases are routed for human review.

## Pipeline

<img width="2528" height="1684" alt="Gemini_Generated_Image_ie6gf8ie6gf8ie6g" src="https://github.com/user-attachments/assets/7a20ff66-5df5-4ef6-b872-9ca9fe11690c" />


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
