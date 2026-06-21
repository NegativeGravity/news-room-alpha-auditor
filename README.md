# News Room Alpha Auditor: Causal NLP & Market-RAG Pipeline

This repository implements a production-grade, end-to-end pipeline for auditing financial news. Engineered for quantitative alpha research, the system rigorously evaluates whether a news headline *actually caused* a market microstructure movement, or if the price action was merely correlation, market noise, or an event that was already fully "priced in."

By fusing CAPM-adjusted market mathematics with a two-stage Retrieval-Augmented Generation (RAG) architecture, this pipeline forces Large Language Models (LLMs) to ground their reasoning in strict financial reality rather than hallucinated narratives.

---

## Architecture Overview

The system is compartmentalized into five core modules, moving from raw tick/news ingestion to final causal verification.

### 1. Data Ingestion & Alignment (`MarketDataLoader`)
Robust loading and chronological alignment of multi-modal datasets:
* **News Data:** Ticker-specific headlines, descriptions, and UTC timestamps.
* **Price Data:** Daily OHLCV (Open, High, Low, Close, Volume) equities data.
* **Index Data:** S&P 500 daily closing prices utilized as the broad market baseline.

### 2. Quantitative Market Metrics (`AdvancedMarketCalculator`)
Before any text processing occurs, the system establishes the mathematical "ground truth" of the asset's behavior. 

* **Pre-event Drift:** The asset's return over the $t-5$ window prior to the news, identifying if the market had already anticipated the catalyst.
* **Volume Z-Score:** A 20-day rolling standard score of trading volume to detect statistically significant market participation:
  $$Z_{volume} = \frac{V_t - \mu_{20}}{\sigma_{20}}$$
* **Dynamic Beta & True Alpha:** Calculates a rolling 60-day covariance with the S&P 500 to establish a dynamic CAPM Beta. The expected market return is subtracted from the asset's actual return to isolate the idiosyncratic **True Alpha**:
  $$\beta_t = \frac{\text{Cov}(R_{stock}, R_{market})}{\text{Var}(R_{market})}$$
  $$\alpha_{true} = R_{stock} - (\beta_t \times R_{market})$$

### 3. NLP Sentiment & Heuristic Labeling (`FinbertLabeler`)
Utilizes the domain-specific `ProsusAI/finbert` model to score headline sentiment (+1, -1, 0). The pipeline then applies a deterministic heuristic ruleset—combining sentiment, True Alpha, and Volume Z-Score—to assign a rigid "Weak Label" (e.g., `SUPPORTED`, `WEAKLY_SUPPORTED`, `UNSUPPORTED`, `UNVERIFIABLE`).

### 4. Two-Stage Market-RAG (`ProFinancialRetriever`)
To provide the auditor with historical context, the pipeline leverages an advanced Retrieval-Augmented Generation setup to surface relevant past news:
* **Bi-Encoder Vector Search:** Uses `BAAI/bge-large-en-v1.5` and `FAISS` inner-product indexing for high-speed, dense candidate retrieval.
* **Cross-Encoder Reranking:** Passes the top candidates through `BAAI/bge-reranker-large` to optimize for maximum contextual relevance before feeding the LLM.

### 5. LLM Causal Auditing & Evaluation (`CausalMarketAuditor` & `ProEvaluator`)
The final verification engine. The LLM is injected with the computed quantitative metrics, the target headline, and the retrieved historical evidence. 

It operates under strict systemic constraints:
1. **Differentiate Correlation from Causation:** Must extract explicit causal connectors (e.g., "due to").
2. **Flag Priced-In Events:** Must reject alpha claims if pre-event drift absorbed the news.
3. **Structured Output:** Forces a strictly typed JSON response containing the verdict, confidence score, and extracted evidence.

The accompanying `ProEvaluator` measures the LLM's causal reasoning against the heuristic baseline, tracking critical system metrics like "Dangerous Hallucinations" and generating formal confusion matrices.
