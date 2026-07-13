"""지휘자가 DB 근거 현황을 확인하는 구조화 도구."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from agent.application.evidence_service import inspect_job_evidence as inspect_evidence
from shared.schema.investigation_schema import EvidenceRequirement, InvestigationConstraints


@tool
def inspect_job_evidence(
    requirements: list[dict],
    constraints: dict | None = None,
) -> str:
    """답변 근거 요구사항별 DB 표본 수, 날짜와 필드 충족도를 확인합니다."""

    import shared.config as config

    report = inspect_evidence(
        config.DB_PATH,
        [EvidenceRequirement.model_validate(item) for item in requirements],
        InvestigationConstraints.model_validate(constraints or {}),
    )
    return json.dumps(report, ensure_ascii=False, indent=2)


__all__ = ["inspect_job_evidence"]
