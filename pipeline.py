import os
from dotenv import load_dotenv
from search_engine import HybridSearchEngine
from reranker import SearchReRanker
from langfuse import get_client, observe

# Load environment variables from .env
load_dotenv()

# Initialize Langfuse client
langfuse = get_client()

class ProductionSearchPipeline:
    def __init__(self):
        print("🚀 Initializing Production Search Pipeline...")
        self.hybrid_engine = HybridSearchEngine()
        self.reranker = SearchReRanker()
        print("✅ Pipeline ready!")

    @observe(name="search_query_execution")
    def search(self, query: str, top_k_hybrid: int = 10, final_top_k: int = 3):
        """
        End-to-end search pipeline wrapped with Langfuse tracing:
        1. Hybrid Retrieval (Vector + BM25 via RRF)
        2. Neural Cross-Encoder Re-ranking
        """
        # Add metadata to the active trace
        langfuse.update_current_trace(
            input=query,
            metadata={"pipeline_version": "v1.0", "search_mode": "hybrid_rerank"}
        )

        # Step 1: Hybrid Retrieval
        print(f"\n🔍 Executing Hybrid Retrieval for: '{query}'")
        hybrid_results = self.hybrid_engine.hybrid_search(query, top_k=top_k_hybrid)
        
        # Step 2: Cross-Encoder Re-ranking
        print(f"🔄 Re-ranking top {len(hybrid_results)} candidates with Cross-Encoder...")
        final_results = self.reranker.rerank(query, hybrid_results, top_k=final_top_k)
        
        # Log final output to trace
        langfuse.update_current_trace(output=final_results)
        
        return final_results

if __name__ == "__main__":
    pipeline = ProductionSearchPipeline()
    query_to_test = "ergonomic office chair"
    
    results = pipeline.search(query_to_test)
    
    print(f"\n🏆 Final Search Results for: '{query_to_test}'\n" + "-"*60)
    for r in results:
        print(f"Rank {r['rank']} | Score: {r['relevance_score']} | Product: {r['title']}")
        
    # Ensure all telemetry logs flush before exiting
    langfuse.flush()
    print("\n📊 Trace successfully sent to Langfuse dashboard!")
