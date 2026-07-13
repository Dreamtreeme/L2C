import json
import asyncio
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from pathlib import Path

from shared.db.database import Database
from agent.application.chat_service import get_chat_service
from agent.application.run_contracts import RunEvent, RunStatus, new_run_id
from agent.application.run_registry import get_run_registry
from agent.utils.logger import logger

app = FastAPI(title="L2C Q&A API Server")

DEFAULT_LOCAL_ORIGINS = (
    "http://127.0.0.1:8000",
    "http://localhost:8000",
)
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", ",".join(DEFAULT_LOCAL_ORIGINS)).split(",")
    if origin.strip()
]
allowed_hosts = [
    host.strip()
    for host in os.getenv(
        "LOCAL_API_ALLOWED_HOSTS",
        "127.0.0.1,localhost,testserver,[::1]",
    ).split(",")
    if host.strip()
]

app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

# CORS 활성화
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or list(DEFAULT_LOCAL_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# 정적 파일 서빙용 디렉토리 생성 보장
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class ChatRequest(BaseModel):
    query: str
    resume_run_id: str | None = None
    conversation_id: str = ""


def _effective_chat_query(
    query: str,
    *,
    resume_run_id: str | None,
    conversation_id: str,
) -> str:
    """확인 질문 또는 최근 대화의 텍스트 문맥을 현재 요청에 결합합니다."""

    registry = get_run_registry()
    if resume_run_id:
        previous = registry.get(resume_run_id)
        previous_result = dict((previous or {}).get("result") or {})
        clarification = dict(previous_result.get("clarification") or {})
        if (
            previous
            and previous.get("status") == RunStatus.WAITING_INPUT.value
            and clarification.get("question")
        ):
            return (
                f"[이전 사용자 요청]\n{previous.get('user_query') or previous.get('query', '')}\n\n"
                f"[지휘자의 확인 질문]\n{clarification['question']}\n\n"
                f"[사용자의 추가 답변]\n{query}"
            )
        if previous and previous.get("status") == RunStatus.CANCELLED.value:
            return (
                f"[취소된 사용자 요청]\n{previous.get('user_query') or previous.get('query', '')}\n\n"
                "[재개 방식]\n오래된 화면 좌표는 재사용하지 말고 현재 화면에서 안전하게 다시 시작하십시오.\n\n"
                f"[사용자의 재개 지시]\n{query}"
            )

    history = get_run_registry().conversation_history(conversation_id, limit=4)
    if not history:
        return query
    turns: list[str] = []
    for item in history:
        result = dict(item.get("result") or {})
        answer = str(result.get("last_action_result") or "").strip()
        if not answer:
            continue
        turns.append(
            f"사용자: {str(item.get('user_query') or item.get('query') or '')[:2000]}\n"
            f"도우미: {answer[:2000]}"
        )
    if not turns:
        return query
    return "[최근 대화 문맥]\n" + "\n\n".join(turns) + f"\n\n[현재 사용자 요청]\n{query}"

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
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": job["id"],
        "company_name": job["company_name"],
        "position": job["position"],
        "url": job["url"],
        "posted_at": job.get("posted_at"),
        "posted_at_text": job.get("posted_at_text"),
        "evidence_hash": job.get("evidence_hash"),
        "collected_at": job["created_at"],
        "raw_text": job["raw_ocr_text"] or f"회사명: {job['company_name']}\n직무: {job['position']}\n기술스택: {job['tech_stack']}"
    }


@app.get("/api/jobs/{job_id}/versions")
async def get_job_versions(job_id: int):
    """공고 내용이 바뀐 시점별 출처 스냅샷을 반환합니다."""
    import shared.config as config

    db = Database(config.DB_PATH)
    if not db.get(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "versions": db.list_versions(job_id)}


@app.get("/api/runs/{run_id}")
async def get_run_status(run_id: str):
    """최근 로컬 요청의 진행 상태와 실행 요약을 반환합니다."""

    item = get_run_registry().get(run_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return item


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """실행 중인 요청이 다음 안전 지점에서 중단되도록 표시합니다."""

    item = get_run_registry().request_cancel(run_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run_id,
        "cancel_requested": bool(item.get("cancel_requested")),
        "status": item.get("status"),
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

        run_id = new_run_id("chat")
        registry = get_run_registry()
        effective_query = _effective_chat_query(
            query,
            resume_run_id=req.resume_run_id,
            conversation_id=req.conversation_id,
        )
        registry.start(
            run_id,
            effective_query,
            conversation_id=req.conversation_id,
            user_query=query,
        )
        event_queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def event_sink(event: RunEvent) -> None:
            registry.apply_event(event)
            loop.call_soon_threadsafe(event_queue.put_nowait, event)

        logger.info("Received query for commander", query=effective_query, run_id=run_id)

        yield f"data: [PROCESSING] {json.dumps({'run_id': run_id}, ensure_ascii=False)}\n\n"

        task = asyncio.create_task(
            asyncio.to_thread(
                get_chat_service().run,
                effective_query,
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
        except asyncio.CancelledError:
            registry.request_cancel(run_id)
            raise
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
                "clarification": result.get("clarification"),
                "resumed_from_run_id": req.resume_run_id,
                "resume_mode": "restart_from_request" if req.resume_run_id else "",
                "conversation_id": req.conversation_id,
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
