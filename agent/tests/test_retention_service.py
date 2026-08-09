import os
import sqlite3
from datetime import datetime, timedelta

from agent.tests.job_test_data import insert_job
from shared.schema.jd_schema import JobCollectionEvidence


def test_operations_api_previews_and_requires_confirmation_header(
    monkeypatch, tmp_path
):
    from fastapi.testclient import TestClient

    from agent.config import get_settings
    from agent.web_server import app

    logs_dir = tmp_path / "logs"
    screenshot_dir = tmp_path / "screenshots"
    logs_dir.mkdir()
    screenshot_dir.mkdir()
    paths = get_settings().paths
    monkeypatch.setattr(paths, "db_path", tmp_path / "operations.db")
    monkeypatch.setattr(paths, "log_dir", logs_dir)
    monkeypatch.setattr(paths, "screenshot_dir", screenshot_dir)

    class FakeChatService:
        def list_runs(self, limit=20):
            return []

    monkeypatch.setattr(
        "agent.web_server._chat_service_for_app",
        lambda _application: FakeChatService(),
    )

    client = TestClient(app)
    preview = client.get("/api/operations")
    denied = client.post("/api/operations/retention")
    applied = client.post(
        "/api/operations/retention",
        headers={"X-L2C-Operation": "apply-retention"},
    )

    assert preview.status_code == 200
    assert "inventory" in preview.json()["retention"]
    assert denied.status_code == 403
    assert applied.status_code == 200
    assert applied.json()["dry_run"] is False


def test_react_ui_sanitizes_output_and_exposes_retention_controls():
    from pathlib import Path

    frontend_src = Path(__file__).resolve().parents[2] / "frontend" / "src"
    message_source = (frontend_src / "components" / "MessageItem.tsx").read_text(
        encoding="utf-8"
    )
    operations_source = (
        frontend_src / "components" / "OperationsDrawer.tsx"
    ).read_text(encoding="utf-8")
    api_source = (frontend_src / "lib" / "api.ts").read_text(encoding="utf-8")

    assert "DOMPurify.sanitize" in message_source
    assert "applyRetention" in operations_source
    assert "X-L2C-Operation" in api_source


def test_retention_is_dry_run_by_default_and_preserves_referenced_artifacts(tmp_path):
    from agent.application.retention_service import RetentionPolicy, run_retention
    from shared.db.database import Database

    now = datetime(2026, 7, 13, 12, 0, 0)
    old = (now - timedelta(days=400)).isoformat(timespec="seconds")
    logs_dir = tmp_path / "logs"
    screenshot_dir = tmp_path / "screenshots"
    logs_dir.mkdir()
    screenshot_dir.mkdir()
    old_log = logs_dir / "old.log"
    old_artifact = screenshot_dir / "old.png"
    referenced_artifact = screenshot_dir / "referenced.png"
    for path in (old_log, old_artifact, referenced_artifact):
        path.write_bytes(b"artifact")
        timestamp = (now - timedelta(days=400)).timestamp()
        os.utime(path, (timestamp, timestamp))

    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    evidence = JobCollectionEvidence(screenshot_path=str(referenced_artifact))
    job_id = insert_job(
        db,
        "https://example.com/jobs/retention",
        {
            "company_name": "Acme",
            "position": "Engineer",
            "content_hash": "retention-job",
            "raw_ocr_text": "version 1",
        },
        evidence=evidence,
    )
    for version in range(2, 8):
        insert_job(
            db,
            "https://example.com/jobs/retention",
            {
                "company_name": "Acme",
                "position": "Engineer",
                "content_hash": "retention-job",
                "raw_ocr_text": f"version {version}",
            },
            evidence=evidence,
        )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE job_versions SET observed_at = ? WHERE job_id = ?", (old, job_id)
        )

    policy = RetentionPolicy(
        log_days=30,
        artifact_days=30,
        job_version_days=180,
        keep_job_versions=5,
    )
    preview = run_retention(
        db_path=db_path,
        logs_dir=logs_dir,
        screenshot_dir=screenshot_dir,
        policy=policy,
        dry_run=True,
        now=now,
    )

    assert preview["files"]["log_count"] == 1
    assert preview["files"]["artifact_count"] == 1
    assert preview["database"]["job_versions"] == 2
    assert old_log.exists() and old_artifact.exists() and referenced_artifact.exists()

    applied = run_retention(
        db_path=db_path,
        logs_dir=logs_dir,
        screenshot_dir=screenshot_dir,
        policy=policy,
        dry_run=False,
        now=now,
    )

    assert applied["dry_run"] is False
    assert not old_log.exists()
    assert not old_artifact.exists()
    assert referenced_artifact.exists()
    assert len(db.list_versions(job_id)) == 5
