import json
import asyncio
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

from shared.db.database import Database
from agent.application.chat_service import get_chat_service
from agent.application.run_contracts import RunEvent, new_run_id
from agent.application.run_registry import get_run_registry
from agent.utils.logger import logger

app = FastAPI(title="L2C Q&A API Server")

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]

# CORS 활성화
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials=("*" not in cors_origins),
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
    import shared.config as config

    db = Database(config.DB_PATH)
    job = db.get(job_id)
    if not job:
        return {"error": "Job not found"}
    return {
        "id": job["id"],
        "company_name": job["company_name"],
        "position": job["position"],
        "url": job["url"],
        "posted_at": job.get("posted_at"),
        "posted_at_text": job.get("posted_at_text"),
        "collected_at": job["created_at"],
        "raw_text": job["raw_ocr_text"] or f"회사명: {job['company_name']}\n직무: {job['position']}\n기술스택: {job['tech_stack']}"
    }


@app.get("/api/runs/{run_id}")
async def get_run_status(run_id: str):
    """최근 로컬 요청의 진행 상태와 실행 요약을 반환합니다."""

    item = get_run_registry().get(run_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return item

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

        run_id = new_run_id("chat")
        registry = get_run_registry()
        registry.start(run_id, query)
        event_queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def event_sink(event: RunEvent) -> None:
            registry.apply_event(event)
            loop.call_soon_threadsafe(event_queue.put_nowait, event)

        logger.info("Received query for commander", query=query, run_id=run_id)

        yield f"data: [PROCESSING] {json.dumps({'run_id': run_id}, ensure_ascii=False)}\n\n"

        task = asyncio.create_task(
            asyncio.to_thread(
                get_chat_service().run,
                query,
                run_id=run_id,
                event_sink=event_sink,
            )
        )
        try:
            while not task.done() or not event_queue.empty():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                payload = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
                yield f"data: [EVENT] {payload}\n\n"

            result = await task
        except Exception as exc:
            registry.fail(run_id, str(exc))
            logger.exception("Commander execution failed", error=str(exc), run_id=run_id)
            error_payload = json.dumps(
                {"run_id": run_id, "message": f"지휘자 에이전트 실행 실패: {exc}"},
                ensure_ascii=False,
            )
            yield f"data: [ERROR] {error_payload}\n\n"
            return

        registry.complete(run_id, result)
        final_payload = json.dumps(
            {
                "run_id": run_id,
                "text": str(result.get("last_action_result") or ""),
                "status": result.get("run_status", "completed"),
                "metrics": result.get("metrics", {}),
            },
            ensure_ascii=False,
        )
        yield f"data: [FINAL] {final_payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
