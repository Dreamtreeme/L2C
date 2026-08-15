"""작업자 그래프가 검토한 공고를 저장 단계에 전달한다."""

from __future__ import annotations

from shared.schema.collection_run import CollectionBatch, PostprocessedCollection


def postprocess_collection_batch(batch: CollectionBatch) -> PostprocessedCollection:
    """검토 완료 결과를 의미 재판정 없이 저장 입력으로 변환한다."""

    return PostprocessedCollection(
        submission=batch.submission,
        collected_jobs=list(batch.collected_jobs),
        rejected_items=list(batch.rejected_items),
        site_name=batch.site_name,
    )


__all__ = ["postprocess_collection_batch"]
