"""Reflex 제출물과 후보 검토용 SQLite 스키마."""

WORKER_SUBMISSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS worker_submissions (
    run_id             TEXT PRIMARY KEY,
    source             TEXT,
    payload_json       TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
"""

RECIPE_CANDIDATES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS recipe_candidates (
    run_id            TEXT PRIMARY KEY,
    status            TEXT NOT NULL DEFAULT 'pending_replay',
    validation_json   TEXT,
    review_attempts   INTEGER NOT NULL DEFAULT 0,
    review_started_at TEXT,
    next_review_at    TEXT,
    review_error      TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES worker_submissions(run_id) ON DELETE CASCADE
);
"""

RECIPE_CANDIDATES_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_recipe_candidates_status ON recipe_candidates(status);",
)

RECIPE_CANDIDATES_QUEUE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_recipe_candidates_review_queue ON recipe_candidates(status, next_review_at, created_at);",
)
