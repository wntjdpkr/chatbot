# app.py - Flask API 서버
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM as Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import torch
import torch.nn.functional as F
import time
import logging

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # CORS 설정으로 Streamlit에서 API 호출 가능

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 설정
os.environ["TORCH_DISABLE_TORCH_SOURCES"] = "1"

# 전역 변수로 모델들 저장
tokenizer = None
emotion_model = None
device = None
label_names = None
vectorstore = None

def load_emotion_model():
    """감정분석 모델 로딩 함수"""
    global tokenizer, emotion_model, device, label_names
    
    try:
        model_dir = "./modified_model"  # safetensors가 포함된 폴더
        logger.info(f"감정분석 모델 로딩 시작: {model_dir}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emotion_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        label_names = emotion_model.config.id2label
        
        logger.info(f"감정분석 모델 로딩 완료 (디바이스: {device})")
        logger.info(f"감정 라벨: {label_names}")
        
    except Exception as e:
        logger.error(f"감정분석 모델 로딩 실패: {str(e)}")
        raise e

def load_vectorstore():
    """RAG용 벡터 DB 로딩 함수"""
    global vectorstore
    
    try:
        logger.info("벡터스토어 로딩 시작")
        
        # RAG 데이터 로딩
        with open("rag_output.json", "r", encoding="utf-8") as f:
            rag_data = json.load(f)
        
        # 텍스트 준비
        texts = [doc["title"] + "\n" + doc["content"] for doc in rag_data]
        logger.info(f"로딩된 문서 수: {len(texts)}")
        
        # 문서 분할
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = splitter.create_documents(texts)
        
        # 임베딩 모델 로딩
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # 벡터스토어 로딩 또는 생성
        persist_dir = "chroma_db"
        if not os.path.exists(persist_dir):
            logger.info("새로운 벡터스토어 생성 중...")
            vectorstore = Chroma.from_documents(
                documents, 
                embeddings, 
                persist_directory=persist_dir
            )
            vectorstore.persist()
        else:
            logger.info("기존 벡터스토어 로딩 중...")
            vectorstore = Chroma(
                persist_directory=persist_dir, 
                embedding_function=embeddings
            )
        
        logger.info("벡터스토어 로딩 완료")
        
    except Exception as e:
        logger.error(f"벡터스토어 로딩 실패: {str(e)}")
        raise e

def analyze_emotion(text):
    """텍스트의 감정 분석 함수"""
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = emotion_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
        
        return predicted_class
    
    except Exception as e:
        logger.error(f"감정 분석 실패: {str(e)}")
        return 0  # 기본값 반환

def get_emotion_instruction(emotion_label):
    """감정에 따른 지시어 매핑 함수"""
    reasoning_map = {
        "슬픔": "슬픔을 느끼는 사용자를 위로하듯 차분하게 사유하며 답변하세요.",
        "분노": "분노의 감정 속에서도 이성적으로 사유하며 차분한 설명을 제공하세요.",
        "기쁨": "기쁜 감정에 공감하며, 더욱 풍부한 통찰을 제공하세요.",
        "두려움": "두려운 감정을 이해하며 용기를 북돋우는 철학적 통찰을 제공하세요.",
        "놀라움": "놀라운 상황에 대해 깊이 있는 철학적 해석을 제공하세요.",
        "혐오": "혐오감을 극복할 수 있는 철학적 관점을 제시하세요.",
        "허무": "허무함에 대해 그 허무를 인간 정신의 성장 기회로 삼을 수 있는 차분한 설명을 제공하세요.",
        "패배/자기혐오": "스스로를 반성하고 재창조할 수 있는 존재로서 인간의 위엄을 상기시켜주세요",
        "절망": "감정을 이해하며 그 절망의 밑바닥에서 새로운 가능성이 움틀 수 있음을 알려주며 차분한 설명을 하세요",
    }
    return reasoning_map.get(emotion_label, f"{emotion_label}의 감정을 고려하여 진지하게 추론해 주세요.")

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인 엔드포인트"""
    try:
        # 모델들이 로딩되었는지 확인
        models_loaded = all([
            tokenizer is not None,
            emotion_model is not None,
            vectorstore is not None
        ])
        
        return jsonify({
            "status": "healthy" if models_loaded else "loading",
            "message": "Flask 서버가 정상 작동 중입니다." if models_loaded else "모델 로딩 중...",
            "models_loaded": models_loaded,
            "device": str(device) if device else "unknown"
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"서버 오류: {str(e)}"
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """채팅 API 엔드포인트 (일반 응답)"""
    try:
        # 요청 데이터 파싱
        data = request.get_json()
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({"error": "메시지가 필요합니다."}), 400
        
        logger.info(f"사용자 입력: {user_input}")
        
        # 감성 분석
        start_time = time.time()
        emotion_id = analyze_emotion(user_input)
        emotion_label = label_names[emotion_id]
        emotion_time = time.time() - start_time
        
        logger.info(f"감정 분석 결과: {emotion_label} (소요시간: {emotion_time:.2f}초)")
        
        # 관련 문서 검색 (RAG)
        start_time = time.time()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        related_docs = retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in related_docs])
        rag_time = time.time() - start_time
        
        logger.info(f"검색된 문서 수: {len(related_docs)} (소요시간: {rag_time:.2f}초)")
        
        # 감정 기반 지시어 매핑
        emotion_instruction = get_emotion_instruction(emotion_label)
        
        # 프롬프트 구성
        prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                f"너는 니체 철학을 기반으로 대답하는 철학자야. "
                f"질문자의 감정은 '{emotion_label}'이며, 다음 지침을 따라야 해: {emotion_instruction}"
            ),
            (
                "system", 
                f"질문과 관련된 문맥 정보:\n\n{context}"
            ),
            ("human", "{question}")
        ])
        
        # LLM 응답 생성
        start_time = time.time()
        llm = Ollama(model="gemma")
        chain = prompt | llm
        
        # 스트림 응답을 문자열로 수집
        response_text = ""
        for chunk in chain.stream({"question": user_input}):
            response_text += chunk
        
        llm_time = time.time() - start_time
        logger.info(f"LLM 응답 생성 완료 (소요시간: {llm_time:.2f}초)")
        
        # 응답 반환
        return jsonify({
            "response": response_text,
            "emotion": emotion_label,
            "emotion_id": emotion_id,
            "context_docs_count": len(related_docs),
            "processing_time": {
                "emotion_analysis": round(emotion_time, 2),
                "rag_search": round(rag_time, 2),
                "llm_generation": round(llm_time, 2)
            }
        })
    
    except Exception as e:
        logger.error(f"채팅 API 오류: {str(e)}")
        return jsonify({"error": f"서버 에러가 발생했습니다: {str(e)}"}), 500

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """스트리밍 응답을 위한 엔드포인트"""
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({"error": "메시지가 필요합니다."}), 400
        
        def generate():
            try:
                # 감성 분석
                emotion_id = analyze_emotion(user_input)
                emotion_label = label_names[emotion_id]
                
                # 관련 문서 검색
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                related_docs = retriever.invoke(user_input)
                context = "\n\n".join([doc.page_content for doc in related_docs])
                
                # 감정 기반 지시어 매핑
                emotion_instruction = get_emotion_instruction(emotion_label)
                
                # 메타데이터 먼저 전송
                metadata = {
                    'type': 'metadata', 
                    'emotion': emotion_label, 
                    'emotion_id': emotion_id,
                    'context_docs_count': len(related_docs)
                }
                yield f"data: {json.dumps(metadata, ensure_ascii=False)}\n\n"
                
                # 프롬프트 구성
                prompt = ChatPromptTemplate.from_messages([
                    (
                        "system", 
                        f"너는 니체 철학을 기반으로 대답하는 철학자야. "
                        f"질문자의 감정은 '{emotion_label}'이며, 다음 지침을 따라야 해: {emotion_instruction}"
                    ),
                    (
                        "system", 
                        f"질문과 관련된 문맥 정보:\n\n{context}"
                    ),
                    ("human", "{question}")
                ])
                
                # LLM 응답 스트리밍
                llm = Ollama(model="gemma")
                chain = prompt | llm
                
                for chunk in chain.stream({"question": user_input}):
                    chunk_data = {
                        'type': 'content', 
                        'content': chunk
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
                # 완료 신호
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                error_data = {
                    'type': 'error', 
                    'error': str(e)
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        
        return Response(generate(), mimetype='text/plain')
    
    except Exception as e:
        logger.error(f"스트리밍 API 오류: {str(e)}")
        return jsonify({"error": f"서버 에러가 발생했습니다: {str(e)}"}), 500

@app.route('/emotions', methods=['GET'])
def get_emotions():
    """사용 가능한 감정 라벨 목록 반환"""
    try:
        return jsonify({
            "emotions": label_names,
            "count": len(label_names)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def initialize_models():
    """서버 시작 시 필요한 모델들을 로딩"""
    try:
        logger.info("=" * 50)
        logger.info("Flask 서버 초기화 시작")
        logger.info("=" * 50)
        
        load_emotion_model()
        load_vectorstore()
        
        logger.info("=" * 50)
        logger.info("모든 모델 로딩 완료! 서버가 준비되었습니다.")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"모델 초기화 실패: {str(e)}")
        raise e

if __name__ == '__main__':
    # 서버 시작 시 모델들 로딩
    initialize_models()
    
    # Flask 서버 실행
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=False,  # 프로덕션에서는 False
        threaded=True
    )
