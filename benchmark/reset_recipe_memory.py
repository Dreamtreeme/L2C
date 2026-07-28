"""채용공고 데이터는 유지하고 Reflex 학습 메모리만 초기화한다."""

from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime
from pathlib import Path


TABLES = (
    "recipe_sources",
    "recipes",
    "recipe_candidates",
    "feedback_episodes",
    "worker_submissions",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    db_path = Path(os.getenv("DB_PATH", "data/jobs.db")).resolve()
    with sqlite3.connect(db_path) as conn:
        existing = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        counts = {
            table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in TABLES
            if table in existing
        }
        print(f"db={db_path}")
        print(f"before={counts}")
        if not args.apply:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db_path.with_name(f"{db_path.stem}_before_recipe_reset_{timestamp}{db_path.suffix}")
        with sqlite3.connect(backup_path) as backup:
            conn.backup(backup)

        conn.execute("BEGIN IMMEDIATE")
        for table in counts:
            conn.execute(f'DELETE FROM "{table}"')
        conn.commit()
        after = {
            table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in counts
        }
        print(f"backup={backup_path}")
        print(f"after={after}")


if __name__ == "__main__":
    main()
