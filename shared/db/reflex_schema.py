"""Shared SQLite schema fragments for reflex feedback/review memory."""

FEEDBACK_EPISODES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feedback_episodes (
    episode_id          TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL,
    run_status          TEXT,
    source              TEXT,
    site                TEXT,
    goal                TEXT,
    page_state_key      TEXT,
    action              TEXT,
    feedback_label      TEXT,
    feedback_reason     TEXT,
    feedback_confidence REAL,
    payload_json        TEXT NOT NULL,
    created_at          TEXT NOT NULL
);
"""

FEEDBACK_EPISODES_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_feedback_run ON feedback_episodes(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_feedback_site_label ON feedback_episodes(site, feedback_label);",
    "CREATE INDEX IF NOT EXISTS idx_feedback_action ON feedback_episodes(action);",
)

WORKER_SUBMISSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS worker_submissions (
    submission_id      TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL,
    source             TEXT,
    site               TEXT,
    goal               TEXT,
    keyword            TEXT,
    run_status         TEXT,
    review_attempt     INTEGER NOT NULL DEFAULT 0,
    review_decision    TEXT,
    review_confidence  REAL,
    feedback_to_worker TEXT,
    payload_json       TEXT NOT NULL,
    review_json        TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
"""

WORKER_SUBMISSIONS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_worker_submissions_run ON worker_submissions(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_worker_submissions_site ON worker_submissions(site, review_decision);",
)

REFLEX_MEMORY_SCHEMA = "\n".join(
    [
        FEEDBACK_EPISODES_TABLE_SQL,
        *FEEDBACK_EPISODES_INDEX_SQL,
        WORKER_SUBMISSIONS_TABLE_SQL,
        *WORKER_SUBMISSIONS_INDEX_SQL,
    ]
)