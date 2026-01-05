import os
import time
import streamlit as st
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from feature_engine import FeatureSearchEngine
from targeting_engine import TargetingEngine

class AppState:
    """Streamlit 세션 상태 키를 상수로 관리합니다."""
    MESSAGES = "messages"
    STEP = "step"
    EXTRACTED_DATA = "extracted_data"
    SELECTED_FEATURES = "selected_features"
    SEGMENTATION_RESULTS = "segmentation_results"
    SCROLL_TRIGGER = "scroll_trigger"

class CampaignExtractor:
    """
    사용자의 자연어 입력에서 마케팅 캠페인 구조를 추출하는 에이전트 클래스입니다.
    """
    
    def __init__(self):
        """환경 설정 및 OpenAI 모델을 초기화합니다."""
        # API Key 확인 (환경변수 또는 Streamlit Secrets)
        api_key = os.getenv("OPENAI_API_KEY")
        try:
            if not api_key and "OPENAI_API_KEY" in st.secrets:
                api_key = st.secrets["OPENAI_API_KEY"]
                os.environ["OPENAI_API_KEY"] = api_key
        except Exception:
            pass
            
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.tz = pytz.timezone("Asia/Seoul")

    def extract(self, user_input: str) -> dict:
        """
        자연어 문장에서 상품명, 일정, 수량, 성공지표 등을 정밀하게 추출합니다.

        Args:
            user_input (str): 사용자의 채팅 메시지

        Returns:
            dict: 추출된 데이터 딕셔너리. 실패 시 None 반환.
        """
        now = datetime.now(self.tz)
        
        system_prompt = f"""
당신은 마케팅 캠페인 전문 분석가입니다. 
사용자의 입력에서 다음 항목을 정밀하게 추출하여 JSON 형식으로만 응답하세요.

- product: 캠페인 대상 상품명
- frequency: 발송 횟수 (정수형 숫자만)
- target_count: 대상 고객 수 (예: 100만, 50,000 등 텍스트 그대로)
- metric: 최우선 성공 지표 (전환율, 클릭률, ROI 등)
- start_days_relative: 시작일이 오늘로부터 며칠 뒤인지 (예: "오늘부터" -> 0, "내일부터" -> 1, "1주일 뒤" -> 7, 없으면 7)
- duration_days: 캠페인 진행 기간 (예: "5일간" -> 5, "1주일 동안" -> 7, 없으면 1)

현재 기준 일자: {now.strftime('%Y-%m-%d')}
"""
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ])
            
            # JSON 응답 파싱
            text = response.content.replace("```json", "").replace("```", "").strip()
            import json
            data = json.loads(text)
            
            # 성공지표 기본값 설정 (입력값이 없으면 '인입률')
            if not data.get("metric") or data.get("metric") == "N/A":
                data["metric"] = "인입률"
                data["metric_defaulted"] = True
            else:
                data["metric_defaulted"] = False

            # 날짜 및 기간 계산
            start_days = data.get("start_days_relative", 7)
            duration = data.get("duration_days", 1)
            
            start_dt = now + timedelta(days=start_days)
            if duration > 1:
                end_dt = start_dt + timedelta(days=duration - 1)
                data["calculated_date"] = f"{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}"
            else:
                data["calculated_date"] = start_dt.strftime("%Y-%m-%d")
            
            return data
        except Exception as e:
            st.error(f"데이터 추출 중 오류가 발생했습니다: {e}")
            return None

def initialize_ui():
    """Streamlit 페이지의 기본 UI와 세션 상태를 초기화합니다."""
    st.set_page_config(page_title="Targeting AI Agent", layout="wide", initial_sidebar_state="collapsed")
    st.title("🎯 Targeting AI 에이전트")
    st.markdown("---")
    
    if AppState.MESSAGES not in st.session_state:
        st.session_state[AppState.MESSAGES] = []
    if AppState.STEP not in st.session_state:
        st.session_state[AppState.STEP] = "input"
    if AppState.EXTRACTED_DATA not in st.session_state:
        st.session_state[AppState.EXTRACTED_DATA] = None
    if AppState.SELECTED_FEATURES not in st.session_state:
        st.session_state[AppState.SELECTED_FEATURES] = None
    if AppState.SEGMENTATION_RESULTS not in st.session_state:
        st.session_state[AppState.SEGMENTATION_RESULTS] = None
    if AppState.SCROLL_TRIGGER not in st.session_state:
        st.session_state[AppState.SCROLL_TRIGGER] = False

def handle_workflow_buttons():
    """워크플로우 버튼(계속, 초기화, 종료)을 처리합니다."""
    col1, col2, col3, _ = st.columns([1.2, 1, 1, 2.8])
    
    # 다음 진행 단계 결정
    if st.session_state[AppState.SELECTED_FEATURES] is None:
        next_step = "discovery"
        button_label = "🚀 유효 피처 검색"
    elif st.session_state[AppState.SEGMENTATION_RESULTS] is None:
        next_step = "segmentation"
        button_label = "🤖 AI 세그먼테이션"
    else:
        next_step = "end" # 더 이상 진행할 단계가 없으면 종료 유도
        button_label = "✅ 프로세스 완료"

    with col1:
        if st.button(button_label, use_container_width=True):
            st.session_state[AppState.STEP] = next_step
            st.rerun()
    with col2:
        if st.button("🔄 초기화", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    with col3:
        if st.button("🛑 종료", use_container_width=True):
            st.session_state[AppState.STEP] = "end"
            st.rerun()

def simulate_progress(label: str, duration: float = 1.0):
    """로딩바를 시뮬레이션합니다."""
    progress_bar = st.progress(0, text=label)
    for i in range(100):
        time.sleep(duration / 100)
        progress_bar.progress(i + 1, text=label)
    time.sleep(0.2)
    progress_bar.empty()

def main():
    """애플리케이션의 메인 제어 흐름을 담당합니다."""
    load_dotenv()
    initialize_ui()
    
    # 엔진 초기화
    extractor = CampaignExtractor()
    search_engine = FeatureSearchEngine()
    targeting_engine = TargetingEngine()

    # 1. 채팅 내역 출력
    for i, msg in enumerate(st.session_state[AppState.MESSAGES]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "table" in msg:
                st.table(msg["table"])
            
            # 마지막 응답이 어시스턴트이고 확인 단계라면 버튼 노출
            if i == len(st.session_state[AppState.MESSAGES]) - 1 and \
               msg["role"] == "assistant" and \
               st.session_state[AppState.STEP] == "confirm":
                handle_workflow_buttons()
    
    # 자동 스크롤을 위한 더미 엘리먼트 및 JS
    st.markdown('<div id="last_elem"></div>', unsafe_allow_html=True)
    if st.session_state[AppState.SCROLL_TRIGGER]:
        st.components.v1.html(
            """
            <script>
                var element = window.parent.document.getElementById('last_elem');
                if (element) {
                    element.scrollIntoView({behavior: 'smooth'});
                }
            </script>
            """,
            height=0,
        )
        st.session_state[AppState.SCROLL_TRIGGER] = False

    # 2. 상태별 비즈니스 로직 처리
    if st.session_state[AppState.STEP] == "end":
        st.success("상담이 종료되었습니다. 이용해 주셔서 감사합니다.")
        if st.button("새로운 상담 시작"):
            st.session_state.clear()
            st.rerun()
        return

    if st.session_state[AppState.STEP] == "discovery":
        with st.chat_message("assistant"):
            simulate_progress("최적의 마케팅 피처를 검색 중입니다...", 1.2)
            results = search_engine.search_and_reason(st.session_state[AppState.EXTRACTED_DATA])
            response_text = "분석 결과, 이번 캠페인에 가장 적합한 **TOP 20 유효 피처** 리스트입니다."
            st.markdown(response_text)
            st.table(results)
            
            # 결과 저장 및 단계 이동
            st.session_state[AppState.SELECTED_FEATURES] = results
            st.session_state[AppState.MESSAGES].append({
                "role": "assistant",
                "content": response_text,
                "table": results
            })
            st.session_state[AppState.STEP] = "confirm"
            st.session_state[AppState.SCROLL_TRIGGER] = True
        st.rerun()

    if st.session_state[AppState.STEP] == "segmentation":
        with st.chat_message("assistant"):
            simulate_progress("머신러닝 기반 랭킹 최적화 및 세그먼테이션을 진행 중입니다...", 1.5)
            results = targeting_engine.process_segmentation(
                st.session_state[AppState.EXTRACTED_DATA],
                st.session_state[AppState.SELECTED_FEATURES]
            )
            response_text = f"랭킹 기반 고객 추출 및 **AI 자동 세그먼테이션**이 완료되었습니다."
            st.markdown(response_text)
            st.table(results)
            
            # 결과 저장 및 단계 이동
            st.session_state[AppState.SEGMENTATION_RESULTS] = results
            st.session_state[AppState.MESSAGES].append({
                "role": "assistant",
                "content": response_text,
                "table": results
            })
            st.session_state[AppState.STEP] = "confirm"
            st.session_state[AppState.SCROLL_TRIGGER] = True
        st.rerun()

    # 3. 사용자 입력 처리 (어떤 단계에서든 입력을 받으면 새로운 시작으로 처리)
    if prompt := st.chat_input("예: 갤럭시26 캠페인을 1주일 뒤에 진행할 건데, 전환율 높은 100만명을 뽑아줘"):
        # 입력이 들어오면 무조건 step을 input으로 돌리고 이전 데이터 초기화 (필요시)
        if st.session_state[AppState.STEP] != "input":
            st.session_state[AppState.STEP] = "input"
            st.session_state[AppState.EXTRACTED_DATA] = None
            st.session_state[AppState.SELECTED_FEATURES] = None
            st.session_state[AppState.SEGMENTATION_RESULTS] = None
            
        st.session_state[AppState.MESSAGES].append({"role": "user", "content": prompt})
        st.session_state[AppState.SCROLL_TRIGGER] = True
            
        with st.chat_message("assistant"):
            simulate_progress("문장을 분석하여 캠페인 요건을 정리 중입니다...", 0.8)
            data = extractor.extract(prompt)
            if data:
                st.session_state[AppState.EXTRACTED_DATA] = data
                # 성공지표 주석 처리 (Markdown 기울임꼴 사용)
                metric_display = data['metric']
                if data.get("metric_defaulted"):
                    metric_display = f"{data['metric']}  *(별도 입력값이 없으면 인입률을 기본으로 설정합니다)*"

                summary = f"""
입력하신 내용을 바탕으로 정리된 **캠페인 세부 요건**입니다:

- **📅 일정**: {data['calculated_date']}
- **📦 상품**: {data['product']}
- **🔄 발송 횟수**: {data['frequency']}회
- **👥 대상 고객**: {data['target_count']}명
- **📈 성공 지표**: {metric_display}

이 정보가 맞다면 **'계속 진행'**을 눌러 유효 피처를 검색해 보세요.
"""
                st.session_state[AppState.MESSAGES].append({"role": "assistant", "content": summary})
                st.session_state[AppState.STEP] = "confirm"
                st.session_state[AppState.SCROLL_TRIGGER] = True
                st.rerun()

if __name__ == "__main__":
    main()
