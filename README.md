# Production AI Search & Evaluation Engine

A production-grade AI search evaluation pipeline featuring hybrid retrieval (Vector + Lexical), cross-encoder re-ranking, OpenTelemetry observability with Langfuse, and automated LLM-as-a-Judge evaluation. Designed as an offline CI/CD test rig for e-commerce search relevance.

## System Architecture
- **Vector Engine**: ChromaDB with  dense embeddings
- **Lexical Engine**: Rank-BM25 sparse keyword search
- **Fusion**: Reciprocal Rank Fusion (RRF) algorithm
- **Re-ranker**: 
- **LLM Evaluator**: Claude () for relevance scoring (0-3)
- **Observability**: Langfuse Cloud tracing & metric scores

## Key Components
1. **Hybrid Retrieval**: Combines sparse keyword search (Rank-BM25) with dense semantic vector search (ChromaDB + all-MiniLM-L6-v2), merged via Reciprocal Rank Fusion (RRF).
2. **Neural Re-Ranking**: Uses  to re-score top candidate matches for precision ranking.
3. **Observability & Tracing**: Integrated with Langfuse Cloud to log traces, execution latency, and evaluation metrics.
4. **LLM-as-a-Judge**: Automated evaluation suite using Claude to score query-item relevance on a 0-3 scale.

## Quickstart
1. **Activate Virtual Environment & Install Dependencies**:
   
2. **Configure Environment Variables**:
   Ensure  contains your , , and .
3. **Run Search Pipeline**:
   
4. **Run Evaluation Suite**:
   