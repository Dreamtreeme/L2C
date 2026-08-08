"""채용공고 필드 계약과 상세 화면별 수집 근거를 관리한다."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent.utils.job_fields import (
    normalize_job_collection_fields,
    required_job_fields,
)
from shared.schema.agent_contract import (
    JOB_COLLECTION_FIELD_LABELS,
    JobCollectionContract,
)
from agent.runtime.worker_contracts import WorkerState


def build_job_collection_contract(
    collection_intent: Mapping[str, Any] | None,
    *,
    profile_fields: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """사이트 기본 필드와 요청 근거 필드를 하나의 계약으로 합친다."""

    contract = JobCollectionContract(
        required_fields=required_job_fields(
            collection_intent,
            profile_fields=profile_fields,
        )
    )
    return contract.model_dump(mode="json")


def required_fields_from_state(state: WorkerState) -> list[str]:
    """그래프 상태에서 현재 실행의 필수 필드 목록을 읽는다."""

    request = state["request"]
    contract = request.get("job_collection_contract")
    if isinstance(contract, Mapping):
        fields = normalize_job_collection_fields(
            contract.get("required_fields")
        )
        if fields:
            return list(fields)
    recipe_params = request.get("recipe_params")
    intent = (
        recipe_params.get("collection_intent")
        if isinstance(recipe_params, Mapping)
        else {}
    )
    return list(required_job_fields(intent if isinstance(intent, Mapping) else {}))


def field_contract_items(fields: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    """프롬프트와 로그에 표시할 표준 키와 한글 라벨을 만든다."""

    return [
        {
            "field": field,
            "label": JOB_COLLECTION_FIELD_LABELS.get(field, field),
        }
        for field in normalize_job_collection_fields(fields)
    ]


def detail_coverage_matches(
    coverage: Mapping[str, Any],
    current_url: str,
    detail_key: str = "",
) -> bool:
    """필드 근거 상태가 현재 상세 화면에 속하는지 판정한다."""

    if not coverage:
        return False
    stored_key = str(coverage.get("detail_key") or "").strip()
    current_key = str(detail_key or "").strip()
    if stored_key and current_key:
        return stored_key == current_key
    return bool(
        current_url
        and str(coverage.get("url") or "").strip() == current_url
    )


def new_job_detail_coverage(
    current_url: str,
    detail_key: str = "",
) -> dict[str, Any]:
    """새 상세 공고에서 사용할 빈 필드 근거 상태를 만든다."""

    return {
        "url": current_url,
        "detail_key": detail_key,
        "field_evidence": {},
        "unavailable_fields": [],
        "page_exhausted": False,
    }


def _short_evidence(value: Any, max_chars: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def _normalize_field_evidence(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    evidence: dict[str, str] = {}
    for field in normalize_job_collection_fields(list(value)):
        text = _short_evidence(value.get(field))
        if text:
            evidence[field] = text
    return evidence


def _system_field_evidence(current_url: str) -> dict[str, str]:
    """브라우저가 확정한 URL만 상세 공고의 시스템 근거로 사용한다."""

    return {"url": current_url} if current_url else {}


def merge_job_detail_coverage(
    coverage: Mapping[str, Any] | None,
    observation: Mapping[str, Any] | None,
    *,
    state: WorkerState,
    current_url: str,
    detail_key: str = "",
) -> dict[str, Any]:
    """현재 화면에서 모델이 판독한 필드 근거를 상세 상태에 누적한다."""

    current = dict(coverage or {})
    if not detail_coverage_matches(current, current_url, detail_key):
        current = new_job_detail_coverage(current_url, detail_key)

    evidence = _normalize_field_evidence(
        current.get("field_evidence")
    )
    # 카드 큐의 회사명·직무명은 탐색 힌트일 뿐 상세 화면의 사실 근거가 아닙니다.
    evidence.update(_system_field_evidence(current_url))

    observed = observation if isinstance(observation, Mapping) else {}
    evidence.update(
        _normalize_field_evidence(observed.get("observed_fields"))
    )

    unavailable = normalize_job_collection_fields(
        observed.get("unavailable_fields")
        if observed.get("page_exhausted") is True
        else current.get("unavailable_fields")
    )
    return {
        "url": current_url,
        "detail_key": detail_key,
        "field_evidence": evidence,
        "unavailable_fields": list(unavailable),
        "page_exhausted": bool(
            current.get("page_exhausted")
            or observed.get("page_exhausted")
        ),
    }


def detail_coverage_status(
    coverage: Mapping[str, Any],
    required_fields: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """필수 필드가 확인·미제공·누락 중 어디에 속하는지 계산한다."""

    required = list(normalize_job_collection_fields(required_fields))
    evidence = _normalize_field_evidence(
        coverage.get("field_evidence")
    )
    found = [field for field in required if field in evidence]
    unavailable = (
        [
            field
            for field in normalize_job_collection_fields(
                coverage.get("unavailable_fields")
            )
            if field in required and field not in found
        ]
        if coverage.get("page_exhausted") is True
        else []
    )
    resolved = set(found) | set(unavailable)
    missing = [field for field in required if field not in resolved]
    return {
        "required_fields": required,
        "found_fields": found,
        "unavailable_fields": unavailable,
        "missing_fields": missing,
        "field_evidence": evidence,
        "page_exhausted": bool(coverage.get("page_exhausted")),
        "complete": not missing,
    }


__all__ = [
    "build_job_collection_contract",
    "detail_coverage_matches",
    "detail_coverage_status",
    "field_contract_items",
    "merge_job_detail_coverage",
    "new_job_detail_coverage",
    "required_fields_from_state",
]
