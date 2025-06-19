# main.py - Streamlit 프론트엔드
import streamlit as st
import requests
import json
import os
import time
from datetime import datetime
from langchain_core.messages import ChatMessage

# Streamlit 페이지 설정 (반드시 맨 처음에 위치)
st.set_page_config(
    page_title="Chat Nietzsche", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.emotion-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    background-color: #f0f2f6;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 0.25rem;
}
.status-healthy { color: #28a745; }
.status-error { color: #dc3545; }
.status-loading { color: #ffc107; }
</style>
""", unsafe_allow_html=True)

# 페이지 제목
st.markdown("""
<div class="main-header">
    <h1>🤖 Chat Gemma - 니체 철학 챗봇</h1>
    <p>감정분석과 RAG를 활용한 니체 철학 기반 AI 챗봇입니다</p>
</div>
""", unsafe_allow_html=True)

# Flask API 서버 설정
FLASK_API_URL = "http://localhost:5000"

# 헬퍼 함수들
def print_messages():
    """이전 대화 메시지들을 출력하는 함수"""
    for message in st.session_state.get("messages", []):
        with st.chat_message(message.role):
            st.write(message.content)

def chat_message_to_dict(msg: ChatMessage):
    """ChatMessage를 딕셔너리로 변환하는 함수"""
    return {
        "role": msg.role, 
        "content": msg.content,
        "timestamp": datetime.now().isoformat()
    }

def save_messages_to_file(messages, file_path):
    """메시지들을 JSON 파일로 저장하는 함수"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                [chat_message_to_dict(m) for m in messages], 
                f, 
                ensure_ascii=False, 
                indent=2
            )
    except Exception as e:
        st.error(f"파일 저장 실패: {str(e)}")

def check_flask_server():
    """Flask 서버 상태를 확인하는 함수"""
    try:
        response = requests.get(f"{FLASK_API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"연결 실패: {str(e)}"}

def call_flask_api(user_input):
    """Flask API를 호출하여 LLM 응답을 받아오는 함수"""
    try:
        response = requests.post(
            f"{FLASK_API_URL}/chat",
            json={"message": user_input},
            timeout=120  # 2분 타임아웃
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API 오류: {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"error": "요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."}
    except requests.exceptions.RequestException as e:
        return {"error": f"API 호출 중 오류가 발생했습니다: {str(e)}"}

def call_flask_api_stream(user_input):
    """Flask API의 스트리밍 엔드포인트를 호출하는 함수"""
    try:
        response = requests.post(
            f"{FLASK_API_URL}/chat/stream",
            json={"message": user_input},
            stream=True,
            timeout=120
        )
        
        if response.status_code == 200:
            return response
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"스트리밍 API 호출 중 오류가 발생했습니다: {str(e)}")
        return None

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 사이드바 설정
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    
    # 세션 ID 입력
    session_id = st.text_input("Session ID", value="default_session")
    json_file_path = f"chat_{session_id}.json"
    
    # 서버 상태 확인
    st.markdown("### 🖥️ 서버 상태")
    server_status = check_flask_server()
    
    if server_status["status"] == "healthy":
        st.markdown('<p class="status-healthy">🟢 서버 연결됨</p>', unsafe_allow_html=True)
        if "models_loaded" in server_status:
            models_status = "✅ 모델 로딩 완료" if server_status["models_loaded"] else "⏳ 모델 로딩 중"
            st.info(models_status)
    elif server_status["status"] == "loading":
        st.markdown('<p class="status-loading">🟡 모델 로딩 중</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-error">🔴 서버 연결 실패</p>', unsafe_allow_html=True)
        st.error(server_status["message"])
        st.info("Flask 서버를 먼저 실행해주세요: `python app.py`")
    
    # 채팅 설정
    st.markdown("### 💬 채팅 설정")
    use_streaming = st.checkbox("스트리밍 응답 사용", value=True, help="실시간으로 응답을 받아볼 수 있습니다")
    show_processing_time = st.checkbox("처리 시간 표시", value=False)
    
    # 대화 기록 관리
    st.markdown("### 📁 대화 기록")
    clear_btn = st.button("🗑️ 대화기록 초기화", use_container_width=True)
    
    if clear_btn:
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []
        if os.path.exists(json_file_path):
            os.remove(json_file_path)
        st.rerun()
    
    # 통계 정보
    if st.session_state.get("messages"):
        user_msg_count = len([m for m in st.session_state['messages'] if m.role == 'user'])
        st.metric("대화 수", user_msg_count)
    
    # 파일 다운로드
    if st.session_state.get("messages") and os.path.exists(json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as f:
            st.download_button(
                label="📥 대화 기록 다운로드",
                data=f.read(),
                file_name=f"chat_history_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # 정보 섹션
    st.markdown("---")
    st.markdown("### ℹ️ 시스템 정보")
    st.markdown(f"**API URL:** `{FLASK_API_URL}`")
    st.markdown("**모델:** Gemma (Ollama)")
    st.markdown("**기능:** 감정분석 + RAG + 니체 철학")
    
    # 감정 정보 (서버에서 가져오기)
    try:
        emotions_response = requests.get(f"{FLASK_API_URL}/emotions", timeout=5)
        if emotions_response.status_code == 200:
            emotions_data = emotions_response.json()
            st.markdown("**감정 라벨:**")
            emotion_labels = emotions_data.get("emotions", {})
            for eid, elabel in emotion_labels.items():
                st.markdown(f'<span class="emotion-badge">{elabel}</span>', unsafe_allow_html=True)
    except:
        pass

# 메인 채팅 영역
col1, col2 = st.columns([3, 1])

with col1:
    # 이전 대화 출력
    print_messages()
    
with col2:
    if st.session_state.get("messages"):
        st.markdown("### 📊 최근 감정")
        # 최근 대화의 감정 정보 표시 (만약 저장되어 있다면)
        pass

# 사용자 입력 처리
if user_input := st.chat_input("무엇이 궁금한가요?"):
    # 서버 상태 재확인
    server_check = check_flask_server()
    if server_check["status"] != "healthy":
        st.error("Flask 서버가 준비되지 않았습니다. 잠시 후 다시 시도해주세요.")
        st.stop()
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        if use_streaming:
            # 스트리밍 응답 처리
            response_stream = call_flask_api_stream(user_input)
            if response_stream:
                response_container = st.empty()
                info_container = st.empty()
                full_response = ""
                emotion_info = ""
                processing_info = {}
                
                for line in response_stream.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            try:
                                data = json.loads(line_str[6:])  # 'data: ' 제거
                                
                                if data['type'] == 'metadata':
                                    emotion_info = data['emotion']
                                    docs_count = data.get('context_docs_count', 0)
                                    info_container.info(f"🎭 감정: {emotion_info} | 📚 참조 문서: {docs_count}개")
                                    
                                elif data['type'] == 'content':
                                    full_response += data['content']
                                    response_container.write(full_response + "▋")
                                    
                                elif data['type'] == 'done':
                                    response_container.write(full_response)
                                    break
                                    
                                elif data['type'] == 'error':
                                    st.error(f"서버 오류: {data['error']}")
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                
                if full_response:
                    st.session_state["messages"].append(ChatMessage(role="assistant", content=full_response))
                    save_messages_to_file(st.session_state["messages"], json_file_path)
            else:
                st.error("스트리밍 응답을 받을 수 없습니다.")
        
        else:
            # 일반 응답 처리
            with st.spinner("🤔 니체가 생각하는 중..."):
                api_response = call_flask_api(user_input)
            
            if "error" in api_response:
                st.error(api_response["error"])
            else:
                # 감정 및 메타 정보 표시
                emotion = api_response.get('emotion', '알 수 없음')
                context_docs_count = api_response.get('context_docs_count', 0)
                
                # 정보 표시
                info_cols = st.columns(3)
                with info_cols[0]:
                    st.info(f"🎭 감정: {emotion}")
                with info_cols[1]:
                    st.info(f"📚 참조 문서: {context_docs_count}개")
                with info_cols[2]:
                    if show_processing_time and 'processing_time' in api_response:
                        total_time = sum(api_response['processing_time'].values())
                        st.info(f"⏱️ 처리시간: {total_time:.1f}초")
                
                # LLM 응답 표시
                assistant_response = api_response.get('response', '')
                st.write(assistant_response)
                
                # 처리 시간 상세 정보 (선택적)
                if show_processing_time and 'processing_time' in api_response:
                    with st.expander("⏱️ 상세 처리 시간"):
                        times = api_response['processing_time']
                        st.json(times)
                
                # 세션에 응답 저장
                st.session_state["messages"].append(ChatMessage(role="assistant", content=assistant_response))
                save_messages_to_file(st.session_state["messages"], json_file_path)

# 하단 정보
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    니체 철학 기반 AI 챗봇 | Flask + Streamlit + Ollama Gemma<br>
    감정분석과 RAG를 통한 개인화된 철학적 대화 경험
</div>
""", unsafe_allow_html=True)