import json
from contextlib import asynccontextmanager
from ipaddress import ip_address
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from agent.bootstrap import ApplicationRuntime
from agent.config import get_settings
from agent.observability.run_contracts import (
    ChatRequest,
    ChatStreamFrame,
)
from shared.db.database import Database


@asynccontextmanager
async def _application_lifespan(application: FastAPI):
    """백엔드가 공유하는 체크포인터, 그래프, 비전 및 후처리 자원을 관리합니다."""

    runtime = ApplicationRuntime(get_settings().paths.db_path)
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


def _is_loopback_client(host: str) -> bool:
    normalized = str(host or "").strip().casefold()
    if normalized in {"testclient", "localhost"}:
        return True
    try:
        return ip_address(normalized.strip("[]")).is_loopback
    except ValueError:
        return False


@app.middleware("http")
async def reject_non_loopback_client(request: Request, call_next):
    """수동 외부 바인딩에서도 원격 제어 요청은 실행하지 않는다."""

    client_host = request.client.host if request.client else ""
    if not _is_loopback_client(client_host):
        return JSONResponse(
            status_code=403,
            content={
                "detail": "L2C API는 이 PC의 loopback 연결만 허용합니다.",
            },
        )
    return await call_next(request)


app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

# CORS 활성화
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or list(DEFAULT_LOCAL_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

frontend_dist_dir = Path(__file__).resolve().parents[1] / "frontend" / "dist"
frontend_index_path = frontend_dist_dir / "index.html"
frontend_assets_dir = frontend_dist_dir / "assets"

if frontend_assets_dir.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(frontend_assets_dir)),
        name="frontend-assets",
    )


@app.get("/")
async def serve_frontend_index():
    if frontend_index_path.is_file():
        return FileResponse(frontend_index_path)
    raise HTTPException(
        status_code=503,
        detail="프론트엔드 빌드가 없습니다. scripts/run_web_app.ps1로 실행하십시오.",
    )


@app.get("/api/jobs/{job_id}")
async def get_job_detail(job_id: int):
    """지정된 job_id 공고의 상세 정보 및 원본 텍스트를 SQLite에서 조회합니다."""
    db = Database(get_settings().paths.db_path)
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
        "source_platform": job.get("source_platform"),
        "raw_text": job["raw_ocr_text"]
        or f"회사명: {job['company_name']}\n직무: {job['position']}\n기술스택: {job['tech_stack']}",
    }


@app.get("/api/runs/{run_id}")
async def get_run_status(run_id: str, request: Request):
    """최근 로컬 요청의 진행 상태와 실행 요약을 반환합니다."""

    item = _chat_service_for_app(request.app).get_run(run_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return item


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str, request: Request):
    """실행 중인 요청이 다음 안전 지점에서 중단되도록 표시합니다."""

    item = _chat_service_for_app(request.app).cancel_run(run_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run_id,
        "cancel_requested": bool(item.get("cancel_requested")),
        "status": item.get("status"),
    }


@app.get("/api/operations")
async def get_operations_summary(request: Request):
    """최근 실행 상태를 반환합니다."""
    return {
        "runs": _chat_service_for_app(request.app).list_runs(limit=20),
    }


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
    """채팅 요청을 애플리케이션 서비스에 전달하고 SSE로 직렬화합니다."""

    async def event_generator():
        service = _chat_service_for_app(request.app)
        async for frame in service.stream(req):
            yield _serialize_chat_frame(frame)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _serialize_chat_frame(frame: ChatStreamFrame) -> str:
    marker = frame.kind.upper()
    if frame.payload is None:
        return f"data: [{marker}]\n\n"
    payload = json.dumps(
        frame.payload.model_dump(mode="json"),
        ensure_ascii=False,
    )
    return f"data: [{marker}] {payload}\n\n"
