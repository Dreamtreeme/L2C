"""수집 결과와 답변의 결정론적 품질 지표를 계산한다."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

from shared.schema.jd_schema import JobPosting


FIELD_ALIASES = {
    "company_name": ("company_name", "회사명"),
    "position": ("position", "직무명", "포지션"),
    "url": ("url", "공고url", "공고_url", "source_url"),
    "main_tasks": ("main_tasks", "주요업무"),
    "requirements": ("requirements", "자격요건"),
    "preferred": ("preferred", "우대사항"),
    "benefits": ("benefits", "혜택", "혜택정보", "혜택 및 복지"),
}
REQUIRED_FIELDS = ("company_name", "position", "url")
CONTENT_FIELDS = ("main_tasks", "requirements", "preferred", "benefits")
RECORD_KEYS = ("jobs", "job_postings", "공고목록", "collected_data")


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


def _field(record: dict[str, Any], name: str) -> Any:
    for alias in FIELD_ALIASES.get(name, (name,)):
        if alias in record:
            return record[alias]
    return None


def normalize_job_record(record: dict[str, Any]) -> dict[str, Any]:
    return {name: _field(record, name) for name in FIELD_ALIASES}


def extract_job_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in RECORD_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if any(alias in payload for aliases in FIELD_ALIASES.values() for alias in aliases):
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


def evaluate_job_records(actual: Any, reference: Any | None = None) -> dict[str, Any]:
    """필수 필드와 URL을 중심으로 수집 결과를 평가한다."""

    actual_records = [normalize_job_record(item) for item in extract_job_records(actual)]
    count = len(actual_records)
    valid_count = 0
    required_present = 0
    content_present = 0
    urls = []
    for record in actual_records:
        try:
            JobPosting.model_validate(record)
            valid_count += 1
        except ValidationError:
            pass
        required_present += sum(_is_present(record.get(field)) for field in REQUIRED_FIELDS)
        content_present += sum(_is_present(record.get(field)) for field in CONTENT_FIELDS)
        normalized_url = _normalized_url(record.get("url"))
        if normalized_url:
            urls.append(normalized_url)

    required_slots = count * len(REQUIRED_FIELDS)
    content_slots = count * len(CONTENT_FIELDS)
    unique_url_rate = len(set(urls)) / len(urls) if urls else 0.0
    result: dict[str, Any] = {
        "record_count": count,
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
    payload = result if isinstance(result, dict) else {}
    target = max(0, int(payload.get("target_count") or 0))
    collected = max(0, int(payload.get("item_count") or payload.get("collected_count") or 0))
    persisted = max(0, int(payload.get("persisted_count") or 0))
    validation = payload.get("persistence_validation") or {}
    persisted_ids = {
        int(item["job_id"])
        for item in validation.get("persisted_items", [])
        if isinstance(item, dict)
        and str(item.get("job_id") or "").isdigit()
        and int(item["job_id"]) > 0
    }
    observed_ids = {
        int(job_id)
        for job_id in (payload.get("observed_job_ids") or [])
        if str(job_id).isdigit() and int(job_id) > 0
    }
    unidentified_persisted = max(0, persisted - len(persisted_ids))
    resolved = len(persisted_ids | observed_ids) + unidentified_persisted
    accepted = (payload.get("review") or {}).get("decision") == "accept"
    target_met = resolved >= target if target else resolved > 0
    return {
        "target_count": target,
        "collected_count": collected,
        "persisted_count": persisted,
        "observed_existing_count": len(observed_ids),
        "resolved_count": resolved,
        "target_fulfillment": round(min(1.0, resolved / target), 6) if target else None,
        "persistence_rate": round(min(1.0, persisted / collected), 6) if collected else 0.0,
        "accepted": accepted,
        "finished": bool(payload.get("is_finished")),
        "passed": bool(accepted and target_met),
    }


def evaluate_answer_citations(
    answer: str,
    valid_ids: list[int],
    expected_ids: list[int] | None = None,
) -> dict[str, Any]:
    citations = [int(value) for value in re.findall(r"\[job_id:(\d+)\]", answer or "")]
    valid = set(valid_ids)
    valid_citations = [value for value in citations if value in valid]
    expected = set(expected_ids or [])
    return {
        "citation_count": len(citations),
        "citation_validity": round(len(valid_citations) / len(citations), 6) if citations else 0.0,
        "expected_citation_coverage": round(len(set(valid_citations) & expected) / len(expected), 6)
        if expected
        else None,
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
