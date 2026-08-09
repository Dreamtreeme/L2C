"""수집 결과와 답변의 결정론적 품질 지표를 계산한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

from agent.utils.job_fields import (
    DETAIL_JOB_FIELDS,
    IDENTITY_JOB_FIELDS,
    normalize_job_collection_fields,
)
from shared.schema.jd_schema import JOB_FIELDS
from shared.schema.jd_schema import JobPosting


REQUIRED_FIELDS = IDENTITY_JOB_FIELDS
CONTENT_FIELDS = DETAIL_JOB_FIELDS


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _normalized_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return ""
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def normalize_job_record(record: dict[str, Any]) -> dict[str, Any]:
    return {name: record.get(name) for name in JOB_FIELDS}


def extract_job_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    value = payload.get("jobs")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if any(field in payload for field in JOB_FIELDS):
        return [payload]
    return []


def _is_present(value: Any) -> bool:
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return bool(str(value or "").strip())


def _strict_value(value: Any, *, url: bool = False) -> Any:
    if isinstance(value, list):
        return [_normalized_text(item) for item in value if _normalized_text(item)]
    return _normalized_url(value) if url else _normalized_text(value)


def evaluate_job_records(
    actual: Any,
    reference: Any | None = None,
    *,
    required_fields: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """필수 필드와 URL을 중심으로 수집 결과를 평가한다."""

    payload_intent = (
        actual.get("collection_intent")
        if isinstance(actual, dict)
        and isinstance(actual.get("collection_intent"), dict)
        else {}
    )
    resolved_required_fields = normalize_job_collection_fields(
        required_fields
        if required_fields is not None
        else payload_intent.get("required_fields")
    )
    if not resolved_required_fields:
        resolved_required_fields = list(REQUIRED_FIELDS)
    raw_actual_records = extract_job_records(actual)
    actual_records = [
        normalize_job_record(item)
        for item in raw_actual_records
    ]
    count = len(actual_records)
    valid_count = 0
    required_present = 0
    unavailable_required = 0
    content_present = 0
    urls = []
    for raw_record, record in zip(
        raw_actual_records,
        actual_records,
        strict=True,
    ):
        try:
            JobPosting.model_validate(record)
            valid_count += 1
        except ValidationError:
            pass
        unavailable = (
            set(
                normalize_job_collection_fields(
                    raw_record.get("_collection_unavailable_fields")
                )
            )
            if raw_record.get("_collection_page_exhausted") is True
            else set()
        )
        for field in resolved_required_fields:
            if _is_present(record.get(field)):
                required_present += 1
            elif field in unavailable:
                required_present += 1
                unavailable_required += 1
        content_present += sum(_is_present(record.get(field)) for field in CONTENT_FIELDS)
        normalized_url = _normalized_url(record.get("url"))
        if normalized_url:
            urls.append(normalized_url)

    required_slots = count * len(resolved_required_fields)
    content_slots = count * len(CONTENT_FIELDS)
    unique_url_rate = len(set(urls)) / len(urls) if urls else 0.0
    result: dict[str, Any] = {
        "record_count": count,
        "required_fields": resolved_required_fields,
        "unavailable_required_field_count": unavailable_required,
        "schema_valid_rate": round(valid_count / count, 6) if count else 0.0,
        "required_field_coverage": round(required_present / required_slots, 6) if required_slots else 0.0,
        "content_field_coverage": round(content_present / content_slots, 6) if content_slots else 0.0,
        "valid_url_count": len(urls),
        "unique_url_rate": round(unique_url_rate, 6),
    }

    reference_records = [normalize_job_record(item) for item in extract_job_records(reference)]
    if reference is None:
        return result

    actual_by_url = {
        _normalized_url(item.get("url")): item
        for item in actual_records
        if _normalized_url(item.get("url"))
    }
    matched = 0
    exact_identity_fields = 0
    exact_content_fields = 0
    compared_content_fields = 0
    for expected in reference_records:
        expected_url = _normalized_url(expected.get("url"))
        observed = actual_by_url.get(expected_url)
        if observed is None:
            continue
        matched += 1
        for field in ("company_name", "position"):
            exact_identity_fields += int(
                _strict_value(observed.get(field)) == _strict_value(expected.get(field))
            )
        for field in CONTENT_FIELDS:
            if not _is_present(expected.get(field)):
                continue
            compared_content_fields += 1
            exact_content_fields += int(
                _strict_value(observed.get(field)) == _strict_value(expected.get(field))
            )

    result["reference"] = {
        "record_count": len(reference_records),
        "url_recall": round(matched / len(reference_records), 6) if reference_records else 0.0,
        "identity_exact_rate": round(exact_identity_fields / (matched * 2), 6) if matched else 0.0,
        "content_exact_rate": round(exact_content_fields / compared_content_fields, 6)
        if compared_content_fields
        else None,
    }
    return result


def evaluate_collection_summary(result: Any) -> dict[str, Any]:
    from agent.runtime.site_context import looks_like_job_detail_url, site_profile_for_url

    payload = result if isinstance(result, dict) else {}
    target = max(0, int(payload.get("target_count") or 0))
    collected = max(0, int(payload.get("collected_count") or 0))
    persisted = max(0, int(payload.get("persisted_count") or 0))
    persisted_items = [
        item
        for item in payload.get("persisted_items", [])
        if isinstance(item, dict)
    ]
    detail_url_items = persisted_items
    valid_detail_urls = 0
    for item in detail_url_items:
        url = str(item.get("url") or "")
        parsed = urlsplit(url)
        if site_profile_for_url(url):
            valid_detail_urls += int(looks_like_job_detail_url(url))
        else:
            valid_detail_urls += int(
                parsed.scheme in {"http", "https"} and bool(parsed.netloc)
            )
    source_url_integrity = (
        valid_detail_urls / len(detail_url_items)
        if detail_url_items
        else 1.0
    )
    observed_ids = {
        int(job_id)
        for job_id in (payload.get("observed_job_ids") or [])
        if str(job_id).isdigit() and int(job_id) > 0
    }
    resolved = max(0, int(payload.get("resolved_count") or 0))
    accepted = payload.get("status") in {"completed", "partial"}
    target_met = resolved >= target if target else resolved > 0
    return {
        "target_count": target,
        "collected_count": collected,
        "persisted_count": persisted,
        "observed_existing_count": len(observed_ids),
        "resolved_count": resolved,
        "target_fulfillment": round(min(1.0, resolved / target), 6) if target else None,
        "persistence_rate": round(min(1.0, persisted / collected), 6) if collected else 0.0,
        "detail_url_checked_count": len(detail_url_items),
        "detail_url_valid_count": valid_detail_urls,
        "source_url_integrity": round(source_url_integrity, 6),
        "accepted": accepted,
        "finished": bool(payload.get("worker_finished")),
        "passed": bool(accepted and target_met and source_url_integrity == 1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deterministic collection quality metrics.")
    parser.add_argument("actual", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    actual = json.loads(args.actual.read_text(encoding="utf-8"))
    reference = (
        json.loads(args.reference.read_text(encoding="utf-8"))
        if args.reference
        else None
    )
    print(json.dumps(evaluate_job_records(actual, reference), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
