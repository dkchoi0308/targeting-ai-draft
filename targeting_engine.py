import os
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import streamlit as st

class TargetingEngine:
    """
    ML/DL 기반의 고객 랭킹 및 AI 세그먼테이션을 수행하는 엔진입니다.
    """

    def __init__(self):
        """
        초기 설정을 수행합니다.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        try:
            if not api_key and "OPENAI_API_KEY" in st.secrets:
                api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass

        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.7, # 창의적인 세그먼트 명칭을 위해 약간의 온도를 높임
            openai_api_key=api_key
        )

    def process_segmentation(self, plan_data: Dict[str, Any], selected_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        고객 랭킹 기반 추출 및 AI 세그먼테이션을 수행합니다.

        Args:
            plan_data (Dict[str, Any]): 캠페인 계획 데이터 (상품명, 발송 횟수, 대상 수 등)
            selected_features (List[Dict[str, Any]]): 선택된 유효 피처 리스트

        Returns:
            List[Dict[str, Any]]: 세그먼테이션 결과 리스트
        """
        # 1. 랭킹 기반 추출 시뮬레이션 (90만 명 등)
        total_target = self._parse_target_count(plan_data.get("target_count", "0"))
        frequency = int(plan_data.get("frequency", 1))
        product = plan_data.get("product", "해당 상품")
        
        # 2. AI를 통한 세그먼트 명칭 및 특징 생성
        segments = self._generate_ai_segments(product, frequency, total_target, selected_features)
        
        return segments

    def _parse_target_count(self, count_str: str) -> int:
        """숫자 문자열에서 정수값을 추출합니다."""
        try:
            # '90만' 등을 '900000'으로 변환하는 간단한 로직
            clean_str = count_str.replace(",", "").replace("명", "").strip()
            if "만" in clean_str:
                return int(float(clean_str.replace("만", "")) * 10000)
            return int(clean_str)
        except:
            return 900000

    def _generate_ai_segments(self, product: str, frequency: int, total_count: int, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        LLM을 활용하여 브랜드, 시장 트렌드, 피처 특성이 통합된 고도화된 세그먼트를 생성합니다.
        """
        # 상위 피처들의 명칭과 근거 요약
        feature_context = []
        for f in features[:5]:
            # '피처명 (세그먼트)' 형태에서 피처명만 추출하거나 전체 활용
            feature_context.append(f"{f['피처명']} (근거: {f['사유'].split('  ')[0]})")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 글로벌 톱티어 브랜드 마케팅 전략가이자 데이터 사이언티스트입니다.
단순한 타겟 분류를 넘어, 브랜드의 유산(Heritage), 시장의 현재 트렌드, 그리고 추출된 데이터 피처의 상관관계를 분석하여 고도화된 세그먼테이션 리포트를 작성하세요.

**수행 가이드:**
1. **수량 및 비중 준수**: 요청받은 발송 횟수(frequency)와 **정확히 일치하는 개수**의 세그먼트를 생성하되, 다음의 비중을 **반드시** 지키세요.
   - **상품 특성 반영 (40%)**: 브랜드 고유의 셀링 포인트(Galaxy AI, iOS 생태계 등)에 결합된 타겟군.
   - **가격 민감도 반영 (30%)**: 구매력, 실속 소비, 프리미엄 선호 등 경제 활동 중심 타겟군.
   - **시장 트렌드 반영 (30%)**: 최신 테크 뉴스, 커뮤니티 이슈 등 외부 트렌드 중심 타겟군.
2. **데이터-기능 연계 분석**: 
   - 각 세그먼트는 상품의 핵심 가치와 피처를 논리적으로 연결해야 합니다.
   - (예: 얼리어답터 지수를 갤럭시 AI 기능이나 신규 출시 뉴스 키워드와 엮어서 설명)
3. **세그먼트 특징(Traits)**: 
   - 할당된 비중(상품/가격/트렌드)에 맞는 전문적인 분석 결과를 기술하세요.
   - **글자수 제한**: 각 세그먼트의 특성은 **최대 3줄(약 100자 내외)**로 핵심만 요약하세요.
4. **세그먼트 명칭(Name)**: 
   - 트렌드와 브랜드 가치가 응축된 '강력한 후킹 포인트' 문구로 작성하세요.
   - **글자수 제한**: 공백 포함 **한글 20자 미만**으로 짧고 강렬하게 작성하세요.

**출력 형식 (JSON 리스트)**:
{{
  "segments": [
    {{
      "name": "후킹 포인트 (20자 미만)",
      "traits": "3줄 이내의 핵심 요약 특징"
    }}
  ]
}}
"""),
            ("user", f"상품명: {product} \n발송 계획: {total_count}명 대상 총 {frequency}회의 서로 다른 세그먼트로 발송 \n핵심 유효 피처: {feature_context} \n\n위 데이터를 기반으로 **반드시 정확히 {frequency}개의 세그먼트**를 생성해줘.")
        ])

        try:
            chain = prompt | self.llm | JsonOutputParser()
            ai_data = chain.invoke({})
            segments_raw = ai_data.get("segments", [])
        except Exception as e:
            # 폴백 로직
            segments_raw = [{"name": f"핵심 타겟 그룹 {i+1}", "traits": f"{product}에 반응도가 높은 핵심 타겟층"} for i in range(frequency)]

        # 정량적 수치 및 날짜 계산 추가
        results = []
        base_date = datetime.now() + timedelta(days=7)
        avg_volume = total_count // frequency
        
        for i, seg in enumerate(segments_raw[:frequency]):
            send_date = base_date + timedelta(days=i*3)
            results.append({
                "세그번호": i + 1,
                "세그명": seg["name"],
                "세그특성": seg["traits"],
                "발송량": f"{avg_volume:,}명",
                "발송일자": send_date.strftime("%Y-%m-%d")
            })
            
        return results
