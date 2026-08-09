"""작업자 상태의 정규화된 공고 목록 연산."""

from __future__ import annotations

from collections.abc import Sequence

from shared.schema.jd_schema import CollectedJob, JobPosting


def job_count(collected_jobs: Sequence[CollectedJob]) -> int:
    return len(collected_jobs)


def job_postings(collected_jobs: Sequence[CollectedJob]) -> list[JobPosting]:
    return [item.posting for item in collected_jobs]


def _job_identity(collected_job: CollectedJob) -> tuple[str, ...]:
    posting = collected_job.posting
    url = str(posting.url or "").strip()
    card_key = collected_job.evidence.source_card_key.strip()
    if url:
        return ("url", url, card_key)
    return (
        "text",
        str(posting.company_name or "").strip(),
        str(posting.position or "").strip(),
    )


def store_collected_job(
    collected_jobs: Sequence[CollectedJob],
    collected_job: CollectedJob,
) -> list[CollectedJob]:
    """같은 공고의 재완료는 교체하고 새 공고는 뒤에 추가한다."""

    identity = _job_identity(collected_job)
    updated = list(collected_jobs)
    for index, existing in enumerate(updated):
        if _job_identity(existing) == identity:
            updated[index] = collected_job
            return updated
    updated.append(collected_job)
    return updated


__all__ = ["job_count", "job_postings", "store_collected_job"]
