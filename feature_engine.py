import os
import random
from typing import List, Dict, Any, Tuple
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st

class FeatureSearchEngine:
    """
    마케팅 피처 검색 및 선정이유(Reasoning) 추출을 담당하는 엔진 클래스입니다.
    
    이 클래스는 FAISS를 사용하여 벡터 검색을 수행하며, 
    OpenAI의 임베딩 모델과 LLM을 활용하여 최적의 피처를 제안합니다.
    """
    
    def __init__(self):
        """
        초기 설정을 수행합니다. 
        API 키 확인, 임베딩 모델 초기화, 피처 데이터 로딩 및 인덱싱을 진행합니다.
        """
        # API Key 확인 (환경변수 또는 Streamlit Secrets)
        api_key = os.getenv("OPENAI_API_KEY")
        try:
            if not api_key and "OPENAI_API_KEY" in st.secrets:
                api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_key=api_key
        )
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0, 
            openai_api_key=api_key
        )
        self.vector_store = None
        self._initialize_features()

    def _initialize_features(self):
        """
        고품질의 고유 피처 셋을 생성하고 FAISS 벡터 스토어를 초기화합니다.
        정량적 지표와 함께 구체적인 행동 근거(URL, 가맹점 등)를 매핑합니다.
        """
        # 카테고리별 고유 피처 및 실제 증거 데이터 정의
        feature_definitions = [
            # Psychographic
            {"cat": "Psychographic (심리/라이프스타일)", "name": "얼리어답터 지수", "unit": "회", "time_unit": "분", "evidence": "GeekNews/Bloter IT 뉴스 구독"},
            {"cat": "Psychographic (심리/라이프스타일)", "name": "해외 트렌드 민감도", "unit": "회", "time_unit": "분", "evidence": "Reddit/Twitch 해외 커뮤니티 접속"},
            {"cat": "Psychographic (심리/라이프스타일)", "name": "가치 소비 성향", "unit": "건", "time_unit": "분", "evidence": "와디즈/텀블벅 펀딩 참여"},
            {"cat": "Psychographic (심리/라이프스타일)", "name": "삼성 브랜드 선호도", "unit": "회", "time_unit": "회", "evidence": "삼성닷컴/삼성멤버스 활동 이력"},
            {"cat": "Psychographic (심리/라이프스타일)", "name": "애플 브랜드 충성도", "unit": "회", "time_unit": "회", "evidence": "애플스토어/Apple 전용 서비스 결제"},
            
            # Behavioral - Purchase
            {"cat": "Behavioral - Purchase (소비 행동)", "name": "식의주 고관여 소비", "unit": "건", "time_unit": "분", "evidence": "마켓컬리 샛별배송 및 무신사 구매"},
            {"cat": "Behavioral - Purchase (소비 행동)", "name": "커피 하이엔드 취향", "unit": "회", "time_unit": "회", "evidence": "스타벅스 리저브/블루보틀 결제"},
            {"cat": "Behavioral - Purchase (소비 행동)", "name": "배달 서비스 의존도", "unit": "회", "time_unit": "분", "evidence": "배달의민족/쿠팡이츠 고빈도 주문"},
            
            # Behavioral - Digital
            {"cat": "Behavioral - Digital (디지털 행동)", "name": "커뮤니티 헤비 유저", "unit": "회", "time_unit": "분/일", "evidence": "에펨코리아/클리앙 체류"},
            {"cat": "Behavioral - Digital (디지털 행동)", "name": "중고거래 액티브 레이팅", "unit": "건", "time_unit": "회", "evidence": "당근마켓 매너온도 및 거래"},
            {"cat": "Behavioral - Digital (디지털 행동)", "name": "숏폼 콘텐츠 소비력", "unit": "회", "time_unit": "분/일", "evidence": "틱톡/유튜브 쇼츠 시청"},
            
            # Finance & Risk
            {"cat": "Finance & Risk (금융/리스크)", "name": "자산 성숙도", "unit": "회 접속", "time_unit": "분", "evidence": "토스/카카오뱅크 자산 연동"},
            {"cat": "Finance & Risk (금융/리스크)", "name": "투자 공격성", "unit": "회 거래", "time_unit": "분", "evidence": "키움증권/미래에셋증권 등 주요 증권사 사이트 접속"},
            
            # Customer Journey
            {"cat": "Customer Journey (고객 여정)", "name": "이탈 조짐 고위험군", "unit": "회", "time_unit": "일", "evidence": "최근 한 달간 앱 미접속"},
            {"cat": "Customer Journey (고객 여정)", "name": "브랜드 옹호자(NPS)", "unit": "회", "time_unit": "분", "evidence": "자발적 상품 후기 작성"}
        ]

        documents = []
        segments = ["서울권", "MZ세대", "직장인", "고소득층", "트렌드세터"]
        
        for i, feat in enumerate(feature_definitions):
            for seg in segments:
                feat_name = f"{feat['name']} ({seg})"
                
                # 정량적 데이터 시뮬레이션
                count_val = random.randint(10, 150)
                time_val = random.randint(20, 300)
                recency_val = random.randint(1, 14)
                
                # 추세 데이터
                trend_w = random.choice(["증가", "유지", "감소"])
                trend_m = random.choice(["증가", "유지", "감소"])
                
                desc = f"{feat['cat']} 분야의 {feat['name']} 지표입니다."
                
                doc = Document(
                    page_content=f"피처명: {feat_name}, 카테고리: {feat['cat']}, 설명: {desc}",
                    metadata={
                        "id": i * len(segments) + segments.index(seg) + 1,
                        "name": feat_name,
                        "category": feat['cat'],
                        "evidence": feat['evidence'],
                        "count": f"{count_val}{feat['unit']}",
                        "time": f"{time_val}{feat['time_unit']}",
                        "recency": f"{recency_val}일 전",
                        "trend_weekly": trend_w,
                        "trend_monthly": trend_m
                    }
                )
                documents.append(doc)

        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def search_and_reason(self, plan_data: dict, k: int = 20) -> List[Dict[str, Any]]:
        """
        추출된 캠페인 데이터를 바탕으로 유사 피처를 검색하고 하이브리드 사유를 생성합니다.
        유사도 점수를 0~1 사이로 정규화하고 내림차순으로 정렬합니다.
        """
        query = f"상품: {plan_data.get('product', '')}, 마케팅 성공 지표: {plan_data.get('metric', '')}"
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_with_scores:
            meta = doc.metadata
            # FAISS L2 거리를 유사도(0~1)로 변환: 1 / (1 + score) 식 사용
            # 점수가 낮을수록(거리가 짧을수록) 1에 수렴함
            similarity = round(1.0 / (1.0 + score), 4)
            reason = self._generate_reasoning(meta, plan_data)
            
            results.append({
                "번호": meta["id"],
                "피처명": meta["name"],
                "카테고리": meta["category"],
                "유사도": similarity,
                "사유": reason
            })
            
        # 유사도 기준 내림차순 정렬
        results.sort(key=lambda x: x["유사도"], reverse=True)
        return results

    def _generate_reasoning(self, feature_meta: dict, plan_data: dict) -> str:
        """
        정량적 지표와 주요 행동 발생처 정보를 결합하여 사유를 생성합니다.
        """
        evidence = feature_meta.get('evidence', '기본 활동 이력')
        count = feature_meta.get('count', '-')
        time = feature_meta.get('time', '-')
        recency = feature_meta.get('recency', '-')
        t_w = feature_meta.get('trend_weekly', '-')
        t_m = feature_meta.get('trend_monthly', '-')
        
        # 하이브리드 사유: 행동 발생처 + 정량 지표 + 추세
        reason = (
            f"📍 **주요 행동 발생처**: `{evidence}`  \n"
            f"📊 **정량 지표**: 발생 {count} / 이용시간 {time} / **최근 {recency} 발생**  \n"
            f"📈 **추세 분석**: 최근 1주일 {t_w} / 최근 1달 {t_m} 추세"
        )
        return reason
