"""지휘자 요청 이해와 계획 품질을 검증하는 대표 사용자 시나리오."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InvestigationScenario:
    scenario_id: str
    query: str
    expected_purpose: str
    expected_clarification_fields: tuple[str, ...] = ()
    optional_clarification_fields: tuple[str, ...] = ()
    expected_count_mode: str = "unspecified"
    expected_evidence_groups: int = 1
    requires_verified_posted_at: bool = False


INVESTIGATION_SCENARIOS = (
    InvestigationScenario(
        "ambiguous_recent_trend",
        "요즘 뜨는 개발자 공고 찾아줘",
        "trend",
        expected_clarification_fields=(
            "recent_period",
            "search_keywords",
            "trend_metric",
        ),
        optional_clarification_fields=("site_scope",),
        expected_count_mode="visible_all",
        expected_evidence_groups=2,
        requires_verified_posted_at=True,
    ),
    InvestigationScenario(
        "backend_unspecified_count",
        "원티드에서 백엔드 개발자 공고 찾아줘",
        "collect",
        expected_count_mode="visible_all",
    ),
    InvestigationScenario(
        "data_engineer_all",
        "원티드에서 데이터 엔지니어 공고를 전부 수집해줘",
        "collect",
        expected_count_mode="visible_all",
    ),
    InvestigationScenario(
        "qa_ten",
        "원티드에서 QA 자동화 엔지니어 공고 10개 찾아줘",
        "collect",
        expected_count_mode="explicit",
    ),
    InvestigationScenario(
        "ai_skill_summary",
        "원티드에서 AI 엔지니어 공고를 모아서 주요 기술을 정리해줘",
        "collect",
        expected_count_mode="visible_all",
    ),
    InvestigationScenario(
        "frontend_backend_compare",
        "원티드에서 프론트엔드와 백엔드 공고를 각각 찾아 비교해줘",
        "compare",
        expected_clarification_fields=("analysis_dimensions",),
        expected_count_mode="visible_all",
        expected_evidence_groups=2,
    ),
    InvestigationScenario(
        "backend_technology_frequency",
        "백엔드 공고에서 가장 자주 요구하는 기술이 뭐야?",
        "lookup",
    ),
    InvestigationScenario(
        "stored_data_experience_compare",
        "저장된 데이터 엔지니어 공고들의 경력 조건을 비교해줘",
        "compare",
    ),
    InvestigationScenario(
        "backend_today",
        "원티드에서 오늘 올라온 백엔드 공고 찾아줘",
        "collect",
        expected_count_mode="visible_all",
        requires_verified_posted_at=True,
    ),
    InvestigationScenario(
        "ai_monthly_growth",
        "원티드에서 지난달보다 AI 개발자 채용이 늘었는지 알려줘",
        "trend",
        expected_clarification_fields=("comparison_period",),
        expected_count_mode="visible_all",
        expected_evidence_groups=2,
        requires_verified_posted_at=True,
    ),
    InvestigationScenario(
        "ios_lookup",
        "iOS 개발자 공고 알려줘",
        "lookup",
        expected_count_mode="visible_all",
    ),
    InvestigationScenario(
        "ios_today_all",
        "원티드에서 오늘 올라온 iOS 공고를 전부 찾아줘",
        "collect",
        expected_count_mode="visible_all",
        requires_verified_posted_at=True,
    ),
)


__all__ = ["INVESTIGATION_SCENARIOS", "InvestigationScenario"]
