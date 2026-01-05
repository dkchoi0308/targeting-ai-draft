import os
from feature_engine import FeatureSearchEngine
from main import CampaignExtractor
from dotenv import load_dotenv

def test_backend():
    load_dotenv()
    print("--- 1. Testing FeatureSearchEngine Initialization ---")
    try:
        engine = FeatureSearchEngine()
        print("Success: FeatureSearchEngine initialized and vector store created.")
    except Exception as e:
        print(f"Error initializing FeatureSearchEngine: {e}")
        return

    print("\n--- 2. Testing CampaignExtractor ---")
    extractor = CampaignExtractor()
    sample_input = "갤럭시26 예약판매 캠페인을 1주일 뒤에 진행할 거야. 대상은 100만 명이고 클릭률을 높이고 싶어."
    try:
        data = extractor.extract(sample_input)
        if data:
            print("Success: Campaign data extracted:")
            for k, v in data.items():
                print(f"  {k}: {v}")
        else:
            print("Error: CampaignExtractor returned None")
            return
    except Exception as e:
        print(f"Error in CampaignExtractor: {e}")
        return

    print("\n--- 3. Testing Semantic Search and Reasoning ---")
    try:
        results = engine.search_and_reason(data, k=5)
        print(f"Success: Found {len(results)} features.")
        for res in results:
            print(f"- [{res['피처명']}] (유사도: {res['유사도']}): {res['사유']}")
    except Exception as e:
        print(f"Error in Search/Reasoning: {e}")

if __name__ == "__main__":
    test_backend()
