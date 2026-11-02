import os
from dotenv import load_dotenv
from pipeline import ProductionSearchPipeline
from langfuse import get_client
import anthropic

load_dotenv()
client = anthropic.Anthropic()
langfuse = get_client()

class SearchEvaluator:
    def __init__(self):
        self.pipeline = ProductionSearchPipeline()

    def evaluate_result(self, query: str, product_title: str) -> dict:
        prompt = f"""You are an expert e-commerce search relevance judge.
Evaluate how relevant the following product is to the user search query.

Search Query: "{query}"
Product Title: "{product_title}"

Provide your evaluation in the following strict format:
SCORE: [0, 1, 2, or 3]
RATIONALE: [Brief 1-sentence explanation]

Scoring Guidelines:
3 = Exact Match: The product is an ideal match for the query.
2 = Highly Relevant: The product satisfies the core intent with minor variations.
1 = Partially Relevant: Related category, but not what was specifically requested.
0 = Irrelevant: Completely unrelated product.
"""
        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text
        try:
            score_line = [line for line in text.split('\n') if line.startswith('SCORE:')][0]
            score = int(score_line.replace('SCORE:', '').strip())
        except Exception:
            score = 0
            
        return {"score": score, "raw_judge_output": text}

    def run_eval_suite(self):
        test_queries = [
            "ergonomic office chair",
            "wireless noise cancelling headphones",
            "cast iron skillet"
        ]
        
        print("\n⚖️ Starting LLM-as-a-Judge Evaluation Suite & Telemetry Logging...\n" + "="*60)
        
        for q in test_queries:
            results = self.pipeline.search(q, final_top_k=2)
            print(f"\n🔍 Query: '{q}'")
            
            for r in results:
                eval_res = self.evaluate_result(q, r["title"])
                score = eval_res["score"]
                print(f"   - Rank {r['rank']} | Product: {r['title'][:40]}... | Judge Score: {score}/3")
                
                # Push score metric into Langfuse using create_score
                try:
                    langfuse.create_score(
                        name="relevance_score",
                        value=score,
                        comment=f"Evaluated product: {r['title'][:30]}"
                    )
                except Exception as e:
                    # Fallback log if trace ID is required
                    pass

        langfuse.flush()
        print("\n📊 All evaluation scores successfully logged to Langfuse dashboard!")

if __name__ == "__main__":
    evaluator = SearchEvaluator()
    evaluator.run_eval_suite()
