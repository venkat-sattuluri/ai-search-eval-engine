from sentence_transformers import CrossEncoder
import torch

class SearchReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"📥 Loading Cross-Encoder re-ranking model ('{model_name}')...")
        # Automatically utilizes Mac Apple Silicon (MPS) if available for hardware acceleration
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)
        print(f"✅ Cross-Encoder loaded successfully on device: {device}")

    def rerank(self, query: str, candidates: list, top_k: int = 3):
        """
        Takes candidate documents from hybrid search and re-scores them using deep cross-attention.
        candidates: list of dicts containing 'product_id', 'title'
        """
        if not candidates:
            return []

        # Prepare pairs of [query, document_text] for the cross-encoder
        pairs = [[query, item["title"]] for item in candidates]
        
        # Compute relevance scores
        scores = self.model.predict(pairs)
        
        # Attach scores back to candidates
        for i, item in enumerate(candidates):
            item["cross_encoder_score"] = float(scores[i])
            
        # Sort by cross-encoder score descending
        reranked = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)
        
        # Slice to final top_k and re-assign clean ranks
        final_results = []
        for rank, item in enumerate(reranked[:top_k]):
            final_results.append({
                "rank": rank + 1,
                "product_id": item["product_id"],
                "title": item["title"],
                "relevance_score": round(item["cross_encoder_score"], 4)
            })
            
        return final_results

if __name__ == "__main__":
    reranker = SearchReRanker()
    sample_query = "wireless noise cancelling headphones"
    sample_candidates = [
        {"product_id": "B00001", "title": "Logitech G Pro X Mechanical Gaming Keyboard RGB"},
        {"product_id": "B00002", "title": "Sony WH-1000XM5 Wireless Industry Leading Noise Canceling Headphones black"},
        {"product_id": "B00003", "title": "Lodge 10.25 Inch Cast Iron Skillet Pre-Seasoned"}
    ]
    
    results = reranker.rerank(sample_query, sample_candidates, top_k=2)
    print(f"\n🔍 Re-ranking Test for Query: '{sample_query}'\n" + "-"*50)
    for r in results:
        print(f"Rank {r['rank']} | Score: {r['relevance_score']} | Product: {r['title']}")
