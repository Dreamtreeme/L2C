import os
import sqlite3
from datetime import datetime, timedelta


def test_retention_defaults_keep_screen_artifacts_as_long_as_audit_history():
    from agent.application.retention_service import RetentionPolicy

    policy = RetentionPolicy()

    assert policy.artifact_days == policy.audit_days == 90


def test_operations_api_previews_and_requires_confirmation_header(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import shared.config as config
    from agent.web_server import app

    logs_dir = tmp_path / "logs"
    screenshot_dir = tmp_path / "screenshots"
    logs_dir.mkdir()
    screenshot_dir.mkdir()
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "operations.db")
    monkeypatch.setattr(config, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(config, "SCREENSHOT_DIR", screenshot_dir)

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


def test_operations_ui_sanitizes_output_and_exposes_retention_controls():
    from pathlib import Path

    html_path = Path(__file__).resolve().parents[1] / "static" / "index.html"
    html = html_path.read_text(encoding="utf-8")

    assert "DOMPurify.sanitize" in html
    assert 'id="ops-panel"' in html
    assert "X-L2C-Operation" in html
    assert 'onclick="showJobDetail' not in html


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
    job_id = db.upsert(
        "https://example.com/jobs/retention",
        {
            "company_name": "Acme",
            "position": "Engineer",
            "content_hash": "retention-job",
            "raw_ocr_text": "version 1",
        },
        screenshot_path=str(referenced_artifact),
    )
    for version in range(2, 8):
        db.upsert(
            "https://example.com/jobs/retention",
            {
                "company_name": "Acme",
                "position": "Engineer",
                "content_hash": "retention-job",
                "raw_ocr_text": f"version {version}",
            },
            screenshot_path=str(referenced_artifact),
        )
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE job_versions SET observed_at = ? WHERE job_id = ?", (old, job_id))
        conn.execute(
            "INSERT INTO feedback_episodes (episode_id, run_id, payload_json, created_at) "
            "VALUES ('old-episode', 'old-run', '{}', ?)",
            (old,),
        )

    policy = RetentionPolicy(
        log_days=30,
        artifact_days=30,
        audit_days=90,
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
    assert preview["database"]["feedback_episodes"] == 1
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
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM feedback_episodes").fetchone()[0] == 0
