import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pathlib import Path

from agent.config import get_settings

from shared.db.database import Database
from agent.application.run_contracts import (
    ChatErrorPayload,
    ChatFinalPayload,
    ChatRequest,
    RunEvent,
    RunStatus,
    new_run_id,
)
from agent.application.run_registry import get_run_registry
from agent.runtime.application_runtime import ApplicationRuntime
from agent.utils.logger import logger
from shared.schema.investigation_schema import ClarificationAnswer


@asynccontextmanager
async def _application_lifespan(application: FastAPI):
    """백엔드가 공유하는 체크포인터, 그래프, 비전 및 후처리 자원을 관리합니다."""

    import shared.config as config

    runtime = ApplicationRuntime(config.DB_PATH)
    application.state.runtime = runtime
    runtime.start()
    try:
        yield
    finally:
        runtime.close(promotion_timeout_sec=0.5)


def _chat_service_for_app(application: FastAPI):
    runtime = getattr(application.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("애플리케이션 런타임이 시작되지 않았습니다.")
    return runtime.chat_service


app = FastAPI(title="L2C Q&A API Server", lifespan=_application_lifespan)

DEFAULT_LOCAL_ORIGINS = (
    "http://127.0.0.1:8000",
    "http://localhost:8000",
)
cors_origins = [
    origin for origin in get_settings().browser.cors_allow_origins if origin
]
allowed_hosts = [
    host for host in get_settings().browser.local_api_allowed_hosts if host
]

app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

# CORS 활성화
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or list(DEFAULT_LOCAL_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-L2C-Operation"],
)

# 정적 파일 서빙용 디렉토리 생성 보장
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

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


@app.get("/api/operations")
async def get_operations_summary():
    """최근 실행 상태와 보존 만료 후보를 반환합니다."""
    import shared.config as config
    from agent.application.retention_service import run_retention

    retention = run_retention(
        db_path=config.DB_PATH,
        logs_dir=config.LOGS_DIR,
        screenshot_dir=config.SCREENSHOT_DIR,
        dry_run=True,
    )
    return {
        "runs": get_run_registry().list_recent(limit=20),
        "retention": retention,
    }


@app.get("/api/contracts")
async def get_backend_contracts():
    """실행 모델에서 생성한 백엔드·에이전트 JSON 계약을 반환합니다."""

    from agent.application.backend_contract import build_backend_contract_manifest

    return build_backend_contract_manifest()


@app.get("/api/taxonomy/stats")
async def get_search_taxonomy_stats():
    """검색 사전 적재 건수와 최상위 직무 카디널리티를 반환합니다."""

    import shared.config as config
    from agent.application.search_taxonomy_import_service import taxonomy_counts
    from agent.application.search_taxonomy_service import SearchTaxonomyService
    from shared.schema.investigation_schema import InvestigationConstraints

    taxonomy = SearchTaxonomyService(config.DB_PATH)
    question = taxonomy.build_domain_question(InvestigationConstraints())
    return {
        "tables": taxonomy_counts(config.DB_PATH),
        "occupation_domains": (
            question.model_dump(mode="json") if question is not None else None
        ),
    }


@app.post("/api/operations/retention")
async def apply_retention(x_l2c_operation: str = Header(default="")):
    """현재 보존 정책의 만료 후보를 실제로 정리합니다."""
    import shared.config as config
    from agent.application.retention_service import run_retention

    if x_l2c_operation != "apply-retention":
        raise HTTPException(status_code=403, detail="Retention confirmation header required")

    return run_retention(
        db_path=config.DB_PATH,
        logs_dir=config.LOGS_DIR,
        screenshot_dir=config.SCREENSHOT_DIR,
        dry_run=False,
    )


@app.post(
    "/api/chat",
    responses={
        200: {
            "description": "SSE frames containing structured progress and final payloads",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
async def chat_endpoint(req: ChatRequest, request: Request):
    """
    지휘자 모델(Commander)에 쿼리를 주입하고 SSE 스트리밍 답변을 전달하는 엔드포인트입니다.
    """
    async def event_generator():
        query = req.query.strip()
        if not query and req.clarification_answer is None:
            yield "data: [ERROR] 질문이 비어있습니다.\n\n"
            return

        run_id = new_run_id("chat")
        registry = get_run_registry()
        investigation_id = str(req.investigation_id or "").strip()
        clarification_answer = req.clarification_answer
        previous = registry.get(req.resume_run_id) if req.resume_run_id else None
        if previous and previous.get("status") == RunStatus.WAITING_INPUT.value:
            previous_result = dict(previous.get("result") or {})
            previous_clarification = dict(previous_result.get("clarification") or {})
            investigation_id = investigation_id or str(
                previous_result.get("investigation_id") or ""
            )
            if clarification_answer is None and previous_clarification.get("question_id"):
                clarification_answer = ClarificationAnswer(
                    question_id=str(previous_clarification["question_id"]),
                    custom_value=query,
                )
        effective_query = (
            query
            if clarification_answer is not None
            else _effective_chat_query(
                query,
                resume_run_id=req.resume_run_id,
                conversation_id=req.conversation_id,
            )
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
                _chat_service_for_app(request.app).run,
                effective_query,
                run_id=run_id,
                event_sink=event_sink,
                conversation_id=req.conversation_id,
                investigation_id=investigation_id,
                clarification_answer=(
                    clarification_answer.model_dump(mode="json")
                    if clarification_answer is not None
                    else None
                ),
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
                ChatErrorPayload(
                    run_id=run_id,
                    message=f"지휘자 에이전트 실행 실패: {exc}",
                ).model_dump(mode="json"),
                ensure_ascii=False,
            )
            yield f"data: [ERROR] {error_payload}\n\n"
            return

        registry.complete(run_id, result)
        final_payload = json.dumps(
            ChatFinalPayload(
                run_id=run_id,
                text=str(result.get("last_action_result") or ""),
                status=result.get("run_status", "completed"),
                clarification=result.get("clarification"),
                investigation_id=result.get("investigation_id", investigation_id),
                resumed_from_run_id=req.resume_run_id,
                resume_mode=(
                    "checkpoint_resume"
                    if investigation_id and clarification_answer is not None
                    else "restart_from_request"
                    if req.resume_run_id
                    else ""
                ),
                conversation_id=req.conversation_id,
                metrics=result.get("metrics", {}),
            ).model_dump(mode="json"),
            ensure_ascii=False,
        )
        yield f"data: [FINAL] {final_payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
