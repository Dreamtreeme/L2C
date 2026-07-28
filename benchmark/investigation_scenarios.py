"""지휘자 요청 이해와 계획 품질을 검증하는 대표 사용자 시나리오."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InvestigationScenario:
    scenario_id: str
    query: str
    expected_purpose: str
    expected_evidence_policy: str = "database_first"
    expected_clarification_fields: tuple[str, ...] = ()
    optional_clarification_fields: tuple[str, ...] = ()
    expected_count_mode: str = "unspecified"
    expected_evidence_groups: int = 1
    expected_occupation_query: str = ""
    expected_collection_search_term: str = ""
    expected_sites: tuple[str, ...] = ()
    expected_evidence_terms: tuple[str, ...] = ()
    expected_scope_required: bool = False
    expected_target_count: int = 0
    requires_verified_posted_at: bool = False
    requires_comparison_dates: bool = False
    requires_assumption: bool = False


INVESTIGATION_SCENARIOS = (
    InvestigationScenario(
        "ambiguous_recent_trend",
        "요즘 뜨는 개발자 공고 찾아줘",
        "collect",
        expected_evidence_policy="web_required",
        expected_clarification_fields=("analysis_dimensions",),
        optional_clarification_fields=("site_scope",),
        expected_count_mode="visible_all",
        expected_evidence_groups=2,
        expected_occupation_query="개발자",
        expected_collection_search_term="개발자",
        requires_verified_posted_at=True,
        requires_assumption=True,
    ),
    InvestigationScenario(
        "backend_unspecified_count",
        "원티드에서 백엔드 개발자 공고 찾아줘",
        "collect",
        expected_evidence_policy="web_required",
        expected_count_mode="visible_all",
        expected_occupation_query="백엔드 개발자",
        expected_collection_search_term="백엔드 개발자",
        expected_sites=("wanted",),
        expected_evidence_terms=("백엔드 개발자",),
    ),
    InvestigationScenario(
        "data_engineer_all",
        "원티드에서 데이터 엔지니어 공고를 전부 수집해줘",
        "collect",
        expected_evidence_policy="web_required",
        expected_count_mode="visible_all",
        expected_occupation_query="데이터 엔지니어",
        expected_collection_search_term="데이터 엔지니어",
        expected_sites=("wanted",),
        expected_evidence_terms=("데이터 엔지니어",),
    ),
    InvestigationScenario(
        "qa_ten",
        "원티드에서 QA 자동화 엔지니어 공고 10개 찾아줘",
        "collect",
        expected_evidence_policy="web_required",
        expected_count_mode="explicit",
        expected_occupation_query="QA 자동화 엔지니어",
        expected_collection_search_term="QA 자동화 엔지니어",
        expected_sites=("wanted",),
        expected_evidence_terms=("QA 자동화 엔지니어",),
        expected_target_count=10,
    ),
    InvestigationScenario(
        "ai_skill_summary",
        "원티드에서 AI 엔지니어 공고를 모아서 주요 기술을 정리해줘",
        "collect",
        expected_evidence_policy="web_required",
        expected_count_mode="visible_all",
        expected_occupation_query="AI 엔지니어",
        expected_collection_search_term="AI 엔지니어",
        expected_sites=("wanted",),
        expected_evidence_terms=("AI 엔지니어",),
    ),
    InvestigationScenario(
        "frontend_backend_compare",
        "원티드에서 프론트엔드와 백엔드 공고를 각각 찾아 비교해줘",
        "compare",
        expected_evidence_policy="web_required",
        expected_count_mode="visible_all",
        expected_evidence_groups=2,
        expected_sites=("wanted",),
    ),
    InvestigationScenario(
        "backend_technology_frequency",
        "백엔드 공고에서 가장 자주 요구하는 기술이 뭐야?",
        "lookup",
        expected_count_mode="visible_all",
        expected_occupation_query="백엔드",
        expected_collection_search_term="백엔드",
        expected_evidence_terms=("백엔드",),
    ),
    InvestigationScenario(
        "stored_data_experience_compare",
        "저장된 데이터 엔지니어 공고들의 경력 조건을 비교해줘",
        "compare",
        expected_evidence_policy="database_only",
        expected_occupation_query="데이터 엔지니어",
        expected_collection_search_term="데이터 엔지니어",
        expected_evidence_terms=("데이터 엔지니어",),
    ),
    InvestigationScenario(
        "backend_today",
        "원티드에서 오늘 올라온 백엔드 공고 찾아줘",
        "collect",
        expected_evidence_policy="web_required",
        expected_count_mode="visible_all",
        expected_occupation_query="백엔드",
        expected_collection_search_term="백엔드",
        expected_sites=("wanted",),
        expected_evidence_terms=("백엔드",),
        requires_verified_posted_at=True,
    ),
    InvestigationScenario(
        "ai_monthly_growth",
        "원티드에서 지난달보다 AI 개발자 채용이 늘었는지 알려줘",
        "trend",
        expected_evidence_policy="web_required",
        expected_count_mode="visible_all",
        expected_evidence_groups=2,
        expected_occupation_query="AI 개발자",
        expected_collection_search_term="AI 개발자",
        expected_sites=("wanted",),
        expected_evidence_terms=("AI 개발자", "AI 개발자"),
        requires_verified_posted_at=True,
        requires_comparison_dates=True,
        requires_assumption=True,
    ),
    InvestigationScenario(
        "ios_lookup",
        "iOS 개발자 공고 알려줘",
        "lookup",
        expected_count_mode="visible_all",
        expected_occupation_query="iOS 개발자",
        expected_collection_search_term="iOS 개발자",
        expected_evidence_terms=("iOS 개발자",),
    ),
    InvestigationScenario(
        "ios_today_all",
        "원티드에서 오늘 올라온 iOS 공고를 전부 찾아줘",
        "collect",
        expected_evidence_policy="web_required",
        expected_count_mode="visible_all",
        expected_occupation_query="iOS",
        expected_collection_search_term="iOS",
        expected_sites=("wanted",),
        expected_evidence_terms=("iOS",),
        requires_verified_posted_at=True,
    ),
)


__all__ = ["INVESTIGATION_SCENARIOS", "InvestigationScenario"]
