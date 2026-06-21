# Newsroom Alpha Auditor 📈🤖

> **From Headlines to Evidence: A Retrieval-Augmented Financial Narrative Auditor for Claim Verification, Market Alignment, and Causal Caution.**

Newsroom Alpha Auditor is an advanced, production-grade AI system...

# Newsroom Alpha Auditor 📈🤖

> **From Headlines to Evidence: A Retrieval-Augmented Financial Narrative Auditor for Claim Verification, Market Alignment, and Causal Caution.**

Newsroom Alpha Auditor is an advanced, production-grade AI system designed to verify financial news narratives. It acts as an evidence-grounded financial fact-checker that prevents LLM hallucination by combining State-of-the-Art natural language processing with hard quantitative market data (CAPM-adjusted alpha and pre-event drift).

## 🏗️ System Architecture

The project is structured into 5 highly decoupled phases:

1. **Advanced Market Evidence Engine:** Calculates True Alpha (using a 60-day rolling covariance/dynamic beta against the S&P 500), Pre-event Drift, and Volume Z-Scores to establish the mathematical reality of a market event. Uses `ProsusAI/finbert` for semantic direction extraction.
2. **Two-Stage Semantic Retrieval (RAG):** Employs a zero-future-leakage architecture. Uses `BAAI/bge-large-en-v1.5` for dense vector retrieval via `FAISS` (Inner Product) and `BAAI/bge-reranker-large` as a cross-encoder to rigorously rerank top candidates based on financial context.
3. **Causal Market Auditor (LLM):** A heavily engineered LLM reasoning pipeline that forces the model to differentiate between correlation and causation, checks for "priced-in" events, and enforces strict JSON schema outputs.
4. **System Evaluator:** Automatically benchmarks the RAG architecture against rule-based and Vanilla LLM baselines using macro F1 scores and confusion matrices.
5. **Failure Analysis:** A critical review module that isolates "Dangerous Hallucinations" (False Positives) and "Missed Signals" (False Negatives) to ensure absolute reliability in trading environments.

## ⚙️ Installation & Setup

```bash
pip install faiss-cpu sentence-transformers transformers torch scikit-learn matplotlib seaborn
