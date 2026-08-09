"""작업자 상태에 정제 전 공고 원문을 누적한다."""

from __future__ import annotations

from collections.abc import Sequence

from shared.schema.jd_schema import JobCapture


def _capture_identity(capture: JobCapture) -> tuple[str, str]:
    return (
        capture.url.strip().rstrip("/"),
        capture.evidence.source_card_key.strip(),
    )


def store_job_capture(
    captures: Sequence[JobCapture],
    capture: JobCapture,
) -> list[JobCapture]:
    """같은 상세 화면을 다시 완료하면 최신 원문과 근거로 교체한다."""

    identity = _capture_identity(capture)
    updated = list(captures)
    for index, existing in enumerate(updated):
        if _capture_identity(existing) == identity:
            updated[index] = capture
            return updated
    updated.append(capture)
    return updated


__all__ = ["store_job_capture"]
