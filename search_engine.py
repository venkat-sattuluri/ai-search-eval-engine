import chromadb
from rank_bm25 import BM25Okapi
import pickle
import pandas as pd

class HybridSearchEngine:
    def __init__(self, chroma_path="./chroma_db", bm25_path="models/bm25_index.pkl"):
        # 1. Load ChromaDB Dense Vector Store
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection(name="amazon_esci_products")
        
        # 2. Load BM25 Sparse Index & Metadata DataFrame
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.df = data["df"]
            
    def vector_search(self, query: str, top_k: int = 10):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        ranked_results = []
        if results and results['ids']:
            for rank, (doc_id, meta) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
                ranked_results.append({
                    "product_id": doc_id,
                    "title": meta.get("product_title", ""),
                    "rank": rank + 1,
                    "source": "dense_vector"
                })
        return ranked_results

    def bm25_search(self, query: str, top_k: int = 10):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        ranked_results = []
        for rank, idx in enumerate(top_indices):
            row = self.df.iloc[idx]
            ranked_results.append({
                "product_id": str(row["product_id"]),
                "title": row["product_title"],
                "rank": rank + 1,
                "source": "sparse_bm25"
            })
        return ranked_results

    def hybrid_search(self, query: str, top_k: int = 5, k: int = 60):
        dense_hits = self.vector_search(query, top_k=top_k*2)
        sparse_hits = self.bm25_search(query, top_k=top_k*2)
        
        # Compute Reciprocal Rank Fusion (RRF) Scores
        rrf_scores = {}
        title_map = {}
        
        for item in dense_hits:
            pid = item["product_id"]
            title_map[pid] = item["title"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (k + item["rank"]))
            
        for item in sparse_hits:
            pid = item["product_id"]
            title_map[pid] = item["title"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (k + item["rank"]))
            
        # Sort products by total RRF score descending
        sorted_pids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        final_results = []
        for rank, (pid, score) in enumerate(sorted_pids):
            final_results.append({
                "rank": rank + 1,
                "product_id": pid,
                "title": title_map.get(pid, ""),
                "rrf_score": round(score, 5)
            })
        return final_results

if __name__ == "__main__":
    engine = HybridSearchEngine()
    test_query = "noise cancelling headphones"
    
    print(f"\n🔍 Testing Hybrid Search Query: '{test_query}'\n" + "-"*50)
    results = engine.hybrid_search(test_query, top_k=3)
    for r in results:
        print(f"Rank {r['rank']} | RRF Score: {r['rrf_score']} | Product: {r['title']}")
