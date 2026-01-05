import os
import random
from feature_engine import FeatureSearchEngine
from dotenv import load_dotenv

def test_enriched_reasoning():
    load_dotenv()
    print("--- Testing Enriched Reasoning and Back-data ---")
    
    # Mock campaign data
    plan_data = {
        "product": "갤럭시26 예약판매",
        "metric": "전환율",
        "target_count": "100만명"
    }
    
    try:
        engine = FeatureSearchEngine()
        results = engine.search_and_reason(plan_data, k=10)
        
        print(f"\n캠페인: {plan_data['product']} / 목표: {plan_data['metric']}")
        print("-" * 50)
        
        for res in results:
            print(f"피처: {res['피처명']}")
            print(f"유형: {res['유형']}")
            print(f"사유: {res['사유']}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_enriched_reasoning()
