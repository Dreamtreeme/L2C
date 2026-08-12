from __future__ import annotations

import sqlite3

from agent.application.clarification_service import apply_clarification_answer
from agent.application.evidence_service import inspect_job_evidence
from agent.application.job_taxonomy_linker import JobTaxonomyLinker
from agent.application.search_constraint_service import SearchConstraintService
from agent.application.search_taxonomy_maintenance import prepare_search_taxonomy
from agent.application.search_taxonomy_service import SearchTaxonomyService
from agent.tests.job_test_data import insert_job
from shared.db.database import Database
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    EvidenceRequirement,
    InvestigationConstraints,
    InvestigationRequest,
)


def _prepared_taxonomy(db_path) -> SearchTaxonomyService:
    return prepare_search_taxonomy(db_path)


def _insert_and_link(
    db: Database,
    taxonomy: SearchTaxonomyService,
    suffix: str,
    **data,
) -> int:
    job_id = insert_job(db, f"https://example.com/jobs/{suffix}", data)
    JobTaxonomyLinker(taxonomy.db_path).link_job(job_id)
    return job_id


def test_dictionary_resolves_occupation_and_skill_aliases(tmp_path):
    taxonomy = _prepared_taxonomy(tmp_path / "jobs.db")

    assert taxonomy.resolve_occupation_concepts("AI 개발자") == [
        "l2c:occupation:ai_ml_engineer"
    ]
    assert taxonomy.resolve_occupation_concepts("iOS") == [
        "l2c:occupation:ios_engineer"
    ]
    assert set(
        taxonomy.resolve_occupation_concepts("AI Agent 개발이나 Python 백엔드")
    ) == {
        "l2c:occupation:ai_agent_engineer",
        "l2c:occupation:backend_engineer",
    }
    assert set(
        taxonomy.resolve_skill_concepts(["LLM 애플리케이션", "Python 백엔드"])
    ) == {
        "l2c:skill:llm",
        "l2c:skill:python",
    }


def test_dictionary_prepare_normalizes_sources_and_keeps_flat_concepts(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    job_id = insert_job(
        db,
        "https://www.wanted.co.kr/wd/123",
        {
            "company_name": "예시회사",
            "position": "iOS 개발자",
            "source_platform": "원티드",
        },
    )
    _prepared_taxonomy(db_path)

    with sqlite3.connect(db_path) as connection:
        concept_types = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT concept_type FROM search_concepts "
                "WHERE status = 'active'"
            )
        }
        relation_table = connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='search_concept_relations'"
        ).fetchone()
        source_platform = connection.execute(
            "SELECT source_platform FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()[0]

    assert concept_types == {"occupation", "skill"}
    assert relation_table is None
    assert source_platform == "Wanted"


def test_missing_occupation_uses_one_direct_question(tmp_path):
    taxonomy = _prepared_taxonomy(tmp_path / "jobs.db")
    service = SearchConstraintService(taxonomy)
    constraints = InvestigationConstraints(occupation_scope_required=True)

    questions = service.prepare_questions(constraints, [])
    investigation = InvestigationRequest(
        investigation_id="clarify-occupation",
        original_query="채용공고 찾아줘",
        constraints=constraints,
        clarification_questions=questions,
    )
    updated = apply_clarification_answer(
        investigation,
        ClarificationAnswer(
            question_id="occupation_query",
            custom_value="AI 에이전트 개발자",
        ),
    )
    enriched = service.enrich(updated.constraints)

    assert len(questions) == 1
    assert questions[0].field == "occupation_query"
    assert enriched.collection_search_term == "AI 에이전트 개발자"
    assert enriched.occupation_concept_keys == [
        "l2c:occupation:ai_agent_engineer"
    ]


def test_evidence_inspection_does_not_reindex_pending_job(tmp_path):
    db_path = tmp_path / "jobs.db"
    taxonomy = _prepared_taxonomy(db_path)
    db = Database(db_path)
    job_id = insert_job(
        db,
        "https://example.com/jobs/pending",
        {
            "company_name": "대기회사",
            "position": "대기 공고",
            "source_platform": "wanted",
        },
    )

    report = inspect_job_evidence(
        db_path,
        [
            EvidenceRequirement(
                requirement_id="pending",
                description="색인 대기 공고",
                scope=InvestigationConstraints(sites=["wanted"]),
            )
        ],
        taxonomy_service=taxonomy,
    )
    with sqlite3.connect(db_path) as connection:
        status = connection.execute(
            "SELECT taxonomy_index_status FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()[0]

    assert report["search_ready_db_rows"] == 0
    assert status == "pending"


def test_dictionary_scope_excludes_unclassified_jobs(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    backend_id = _insert_and_link(
        db,
        taxonomy,
        "backend",
        position="백엔드 개발자",
        job_category="백엔드 개발",
    )
    _insert_and_link(
        db,
        taxonomy,
        "unclassified",
        position="서비스 운영 담당자",
        job_category="운영",
    )
    requirement = EvidenceRequirement(
        requirement_id="backend",
        description="백엔드 공고",
        scope=InvestigationConstraints(
            occupation_query="백엔드 개발자",
            occupation_concept_keys=["l2c:occupation:backend_engineer"],
        ),
    )

    report = inspect_job_evidence(db_path, [requirement])

    assert report["requirements"][0]["document_ids"] == [backend_id]
    assert report["requirements"][0]["semantic_review_required"] is False


def test_semantic_filter_forces_body_validation_after_dictionary_match(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    job_id = _insert_and_link(
        db,
        taxonomy,
        "backend-with-degree",
        position="백엔드 개발자",
        experience_min=2,
        experience_max=5,
        experience_text="경력 2~5년",
        education="4년제 관련 전공",
    )
    requirement = EvidenceRequirement(
        requirement_id="backend",
        description="학력 조건을 적용한 백엔드 공고",
        scope=InvestigationConstraints(
            occupation_query="백엔드 개발자",
            occupation_concept_keys=["l2c:occupation:backend_engineer"],
            semantic_filters=["4년제 관련 전공 학위가 필수인 공고 제외"],
        ),
    )

    report = inspect_job_evidence(db_path, [requirement])

    requirement_report = report["requirements"][0]
    assert requirement_report["document_ids"] == [job_id]
    assert requirement_report["semantic_review_required"] is True
    assert requirement_report["candidates"][0]["experience_min"] == 2


def test_required_experience_ceiling_filters_structured_candidates(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    included_id = _insert_and_link(
        db,
        taxonomy,
        "included-experience",
        position="AI 에이전트 개발자",
        experience_min=3,
    )
    _insert_and_link(
        db,
        taxonomy,
        "over-experience",
        position="AI 에이전트 개발자",
        experience_min=4,
    )
    _insert_and_link(
        db,
        taxonomy,
        "unknown-experience",
        position="AI 에이전트 개발자",
    )

    report = inspect_job_evidence(
        db_path,
        [
            EvidenceRequirement(
                requirement_id="agent",
                description="주니어 공고",
                scope=InvestigationConstraints(
                    maximum_required_experience_years=3
                ),
            )
        ],
    )

    assert report["document_ids"] == [included_id]


def test_occupation_or_skill_mode_unions_candidate_groups(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    occupation_id = _insert_and_link(
        db,
        taxonomy,
        "backend-occupation",
        position="백엔드 개발자",
    )
    skill_id = _insert_and_link(
        db,
        taxonomy,
        "python-skill",
        position="서비스 운영 담당자",
        requirements=["Python 사용 경험"],
    )
    requirement = EvidenceRequirement(
        requirement_id="backend-or-python",
        description="백엔드 또는 Python 공고",
        scope=InvestigationConstraints(
            occupation_concept_keys=["l2c:occupation:backend_engineer"],
            skill_concept_keys=["l2c:skill:python"],
            occupation_skill_match_mode="any",
        ),
    )

    report = inspect_job_evidence(db_path, [requirement])

    assert report["document_ids"] == [occupation_id, skill_id]


def test_skill_links_preserve_required_and_preferred_sections(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    job_id = _insert_and_link(
        db,
        taxonomy,
        "ios-skills",
        position="iOS 개발자",
        requirements=["SwiftUI 사용 경험"],
        preferred=["RxSwift 경험"],
        main_tasks=["Flutter 앱 유지보수"],
    )

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT concepts.concept_key, links.evidence_field,
                   links.requirement_type
            FROM job_concept_links AS links
            JOIN search_concepts AS concepts ON concepts.id = links.concept_id
            WHERE links.job_id = ? AND links.link_type = 'skill'
            ORDER BY concepts.concept_key, links.evidence_field
            """,
            (job_id,),
        ).fetchall()

    assert ("l2c:skill:swiftui", "requirements", "required") in rows
    assert ("l2c:skill:rxswift", "preferred", "preferred") in rows
    assert ("l2c:skill:flutter", "main_tasks", "mentioned") in rows
    assert taxonomy.matching_skill_job_ids(
        ["l2c:skill:swiftui"],
        requirement_type="required",
    ) == {job_id}


def test_main_tasks_create_semantic_occupation_candidate(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    job_id = _insert_and_link(
        db,
        taxonomy,
        "product-agent",
        company_name="제품회사",
        position="Product Engineer",
        main_tasks=["AI Agent 기반 브라우저 자동화 기능 개발"],
    )
    requirement = EvidenceRequirement(
        requirement_id="agent-role",
        description="AI Agent 개발 공고",
        scope=InvestigationConstraints(
            occupation_query="AI Agent 개발",
            occupation_concept_keys=["l2c:occupation:ai_agent_engineer"],
        ),
    )

    report = inspect_job_evidence(db_path, [requirement])

    assert taxonomy.matching_occupation_job_ids(
        ["l2c:occupation:ai_agent_engineer"]
    ) == {job_id}
    requirement_report = report["requirements"][0]
    assert requirement_report["document_ids"] == [job_id]
    assert requirement_report["semantic_review_required"] is True


def test_reindex_is_idempotent_and_preserves_source_tech_stack(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    job_id = _insert_and_link(
        db,
        taxonomy,
        "idempotent",
        position="AI 엔지니어",
        tech_stack=["Python"],
        requirements=["AWS와 FastAPI 경험"],
        preferred=["Docker 경험"],
    )

    first = JobTaxonomyLinker(taxonomy.db_path).relink_all_jobs()
    with sqlite3.connect(db_path) as connection:
        first_links = connection.execute(
            "SELECT COUNT(*) FROM job_concept_links WHERE job_id = ?",
            (job_id,),
        ).fetchone()[0]
    second = JobTaxonomyLinker(taxonomy.db_path).relink_all_jobs()
    with sqlite3.connect(db_path) as connection:
        second_links = connection.execute(
            "SELECT COUNT(*) FROM job_concept_links WHERE job_id = ?",
            (job_id,),
        ).fetchone()[0]

    assert first_links == second_links
    assert first["skills"] == second["skills"]
    assert Database(db_path).get(job_id)["tech_stack"] == ["Python"]


def test_failed_taxonomy_index_is_retried_once_before_search(tmp_path):
    db_path = tmp_path / "retry-index.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    job_id = insert_job(
        db,
        "https://example.com/jobs/retry-index",
        {
            "company_name": "예시회사",
            "position": "AI 엔지니어",
            "job_category": "AI 엔지니어",
            "requirements": ["Python"],
        },
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET taxonomy_index_status = 'failed',
                taxonomy_index_attempts = 1,
                taxonomy_index_error = 'temporary failure'
            WHERE id = ?
            """,
            (job_id,),
        )
        connection.commit()

    retry = JobTaxonomyLinker(taxonomy.db_path).relink_pending_jobs(
        limit=10,
        max_attempts=2,
    )

    with sqlite3.connect(db_path) as connection:
        status = connection.execute(
            """
            SELECT taxonomy_index_status, taxonomy_index_attempts,
                   taxonomy_index_error
            FROM jobs WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
    assert retry["indexed"] == 1
    assert status == ("indexed", 2, None)
    assert taxonomy.matching_occupation_job_ids(
        ["l2c:occupation:ai_ml_engineer"]
    ) == {job_id}
    assert JobTaxonomyLinker(taxonomy.db_path).relink_pending_jobs(
        limit=10,
        max_attempts=2,
    )["jobs"] == 0
