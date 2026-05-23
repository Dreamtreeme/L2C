import json
import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

from shared.config import DB_PATH
from shared.db.database import Database
from agent.tools.rag_engine import parse_filters, retrieve
from agent.graph.nodes import validate_citations
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="L2C RAG Q&A API Server")

# CORS 활성화
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙용 디렉토리 생성 보장
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class ChatRequest(BaseModel):
    query: str

@app.get("/")
async def redirect_to_index():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

@app.get("/api/jobs/{job_id}")
async def get_job_detail(job_id: int):
    """지정된 job_id 공고의 상세 정보 및 원본 텍스트를 SQLite에서 조회합니다."""
    db = Database(DB_PATH)
    job = db.get(job_id)
    if not job:
        return {"error": "Job not found"}
    return {
        "id": job["id"],
        "company_name": job["company_name"],
        "position": job["position"],
        "url": job["url"],
        "collected_at": job["created_at"],
        "raw_text": job["raw_ocr_text"] or f"회사명: {job['company_name']}\n직무: {job['position']}\n기술스택: {job['tech_stack']}"
    }

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    RAG 검색 및 Gemini 스트리밍 답변 출력을 수행하는 SSE 엔드포인트입니다.
    """
    async def event_generator():
        query = req.query.strip()
        if not query:
            yield "data: [ERROR] 질문이 비어있습니다.\n\n"
            return

        logger.info(f"Received query: {query!r}")

        # 1. 정규식 쿼리 필터 파싱
        filters = parse_filters(query)
        
        # 2. RAG 검색 및 코사인 유사도 연산
        retrieved_data = retrieve(query, filters, DB_PATH)
        
        # 3. Pre-LLM Check
        if retrieved_data.get("is_empty", True) or not retrieved_data.get("results"):
            logger.info("Pre-LLM Check Rejection: Sending rejection text and closing.")
            yield "data: 수집된 공고 내에서 조건에 맞는 정보를 찾을 수 없습니다.\n\n"
            yield "data: [DONE]\n\n"
            return
            
        results = retrieved_data["results"]
        valid_ids = [item["id"] for item in results]
        logger.info(f"Search candidates selected: {valid_ids}")
        
        # 4. XML 앵커링 컨텍스트 생성
        context_parts = []
        for item in results:
            doc_xml = (
                f'<document id="{item["id"]}">\n'
                f'  <source_url>{item["url"]}</source_url>\n'
                f'  <company>{item["company_name"]}</company>\n'
                f'  <position>{item["position"]}</position>\n'
                f'  <content>\n{item["raw_text"]}\n  </content>\n'
                f'</document>'
            )
            context_parts.append(doc_xml)
        context_str = "\n\n".join(context_parts)
        
        # 5. 엄격한 컨텍스트 밀착 프롬프트
        system_prompt = (
            "당신은 채용 공고 분석을 전담하는 Q&A 에이전트입니다. 아래 지침을 엄격히 준수하여 답변하십시오.\n\n"
            "지침:\n"
            "1. 제시된 <document> 태그 내부의 내용만을 근거로 답변하고, 외부 지식이나 상식을 절대 섞지 마십시오.\n"
            "2. 답변의 모든 주장이나 사실적 진술 뒤에는 해당 정보의 출처가 되는 문서의 ID를 반드시 [job_id:ID] 형태로 표기하십시오.\n"
            "   예: '로이드케이에서는 SwiftUI 경험을 우대합니다 [job_id:1]'\n"
            "3. 만약 제공된 정보 내에 답변의 근거가 전혀 존재하지 않거나 부족하다면, 자의적으로 상상하여 답변하지 말고 "
            "   반드시 '공고에서 확인되지 않음' 또는 '수집된 공고 내에서 조건에 맞는 정보를 찾을 수 없습니다.'라고만 대답하십시오.\n"
            "4. 추측이나 주관적인 보완 진술은 일절 허용되지 않습니다."
        )
        
        human_prompt = (
            f"질문: {query}\n\n"
            f"제시된 문서 컨텍스트:\n{context_str}\n\n"
            "위 문서를 바탕으로 질문에 정확하게 답변하십시오. 답변에 인용 ID [job_id:ID]를 명시하지 못할 경우 해당 내용은 삭제되어야 합니다."
        )
        
        qa_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        full_answer_list = []
        try:
            logger.info("Starting stream generation...")
            for chunk in qa_llm.stream(messages):
                content = chunk.content
                full_answer_list.append(content)
                # 클라이언트 단으로 실시간 전송
                yield f"data: {content}\n\n"
                await asyncio.sleep(0.01) # 이벤트 양보
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            yield f"data: [ERROR] 답변 생성 실패: {str(e)}\n\n"
            return
            
        full_answer = "".join(full_answer_list)
        # 6. 인용 ID 검증 및 보정 치환
        final_answer = validate_citations(full_answer, valid_ids)
        logger.info("Stream generation finished. Attributed citations verified.")
        
        # 7. 보정 완료된 최종 답변 페이로드 쏭
        final_payload = json.dumps({"text": final_answer})
        yield f"data: [FINAL] {final_payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
