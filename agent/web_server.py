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
from agent.graph.nodes import qa_reasoning_node
from agent.graph.state import GraphState

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
    지휘자 모델(Commander)에 쿼리를 주입하고 SSE 스트리밍 답변을 전달하는 엔드포인트입니다.
    """
    async def event_generator():
        query = req.query.strip()
        if not query:
            yield "data: [ERROR] 질문이 비어있습니다.\n\n"
            return

        logger.info(f"Received query for commander: {query!r}")

        # 1. 수신 즉시 클라이언트에 확인 응답 — realtime_scraping 트리거 시 수 분 공백 방지
        yield "data: [PROCESSING]\n\n"

        # 2. 지휘자 에이전트 노드 비동기 실행 (RAG 및 실시간 스크래핑을 지휘자가 직접 조율)
        state = GraphState(
            goal=query,
            ui_context="",
            current_markers=[],
            action_history=[],
            recent_images=[],
            marked_image="",
            error_count=0,
            is_finished=False,
            collected_data=[],
            extracted_jd={},
            last_action_result=None,
            plan=[],
            current_plan_step=0,
            step_durations=[],
        )
        try:
            result = await asyncio.to_thread(qa_reasoning_node, state)
            final_answer = result.get("last_action_result", "")
        except Exception as e:
            logger.error(f"Commander execution failed: {e}")
            yield f"data: [ERROR] 지휘자 에이전트 실행 실패: {str(e)}\n\n"
            return

        # 3. 클라이언트 단으로 타이핑 효과 실시간 스트리밍
        logger.info("Streaming commander's final answer to client...")
        for char in final_answer:
            yield f"data: {char}\n\n"
            await asyncio.sleep(0.01) # 10ms 지연으로 자연스러운 타이핑 UX 제공

        # 3. 보정 완료된 최종 답변 페이로드 전달
        final_payload = json.dumps({"text": final_answer})
        yield f"data: [FINAL] {final_payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
