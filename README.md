# Production AI Search & Evaluation Engine

A production-grade AI search evaluation pipeline featuring hybrid retrieval (Vector + Lexical), cross-encoder re-ranking, OpenTelemetry observability with Langfuse, and automated LLM-as-a-Judge evaluation.

## System Architecture

- **Vector Engine**: ChromaDB with `all-MiniLM-L6-v2` dense embeddings
- **Lexical Engine**: Rank-BM25 sparse keyword search
- **Fusion**: Reciprocal Rank Fusion (RRF) algorithm
- **Re-ranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM Evaluator**: Claude (`claude-haiku-5-5`) for relevance scoring (0–3)
- **Observability**: Langfuse Cloud tracing & metric scores

## Quickstart

1. **Activate Virtual Environment & Install Dependencies**:
   source venv/bin/activate
   pip install -r requirements.txt

2. **Configure Environment Variables**:
   Ensure `.env` contains your active `ANTHROPIC_API_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY`.

3. **Run Pipeline Execution**:
   python pipeline.py

4. **Run Automated Evaluation Suite**:
   python eval_judge.py
