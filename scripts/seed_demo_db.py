"""버전 관리되는 합성 공고로 재현 가능한 데모 DB를 생성합니다."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.application.job_taxonomy_linker import JobTaxonomyLinker
from agent.application.search_taxonomy_import_service import import_local_seed
from shared.db.database import Database
from shared.schema.jd_schema import JobPosting


DEMO_DB = ROOT / "data" / "demo_jobs.db"
DEMO_JOBS = ROOT / "data" / "samples" / "demo_jobs.json"
TAXONOMY_SEED = ROOT / "data" / "samples" / "search_taxonomy_ko.json"


def _remove_previous_database() -> None:
    for suffix in ("", "-shm", "-wal"):
        path = Path(f"{DEMO_DB}{suffix}")
        if path.exists():
            path.unlink()


def main() -> int:
    _remove_previous_database()
    postings = json.loads(DEMO_JOBS.read_text(encoding="utf-8"))

    database = Database(DEMO_DB)
    import_local_seed(DEMO_DB, TAXONOMY_SEED)
    linker = JobTaxonomyLinker(DEMO_DB)

    job_ids: list[int] = []
    for payload in postings:
        job_id = database.upsert(JobPosting.model_validate(payload))
        linker.link_job(job_id)
        job_ids.append(job_id)

    print(
        json.dumps(
            {
                "database": str(DEMO_DB),
                "job_count": len(job_ids),
                "job_ids": job_ids,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
