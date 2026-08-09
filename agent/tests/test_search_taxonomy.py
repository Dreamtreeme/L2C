from __future__ import annotations

import csv
import io
import json
import sqlite3
import zipfile

from agent.application.clarification_service import apply_clarification_answer
from agent.application.evidence_service import inspect_job_evidence
from agent.application.job_taxonomy_linker import JobTaxonomyLinker
from agent.application.search_taxonomy_import_service import import_onet_archive
from agent.application.search_taxonomy_maintenance import prepare_search_taxonomy
from agent.application.search_taxonomy_question_builder import TaxonomyQuestionBuilder
from agent.application.search_taxonomy_review_service import SearchTaxonomyReviewService
from agent.application.search_taxonomy_service import (
    DEFAULT_LOCAL_SEED,
    SearchTaxonomyService,
)
from agent.graph.investigation_context import InvestigationModels
from agent.tests.job_test_data import insert_job
from shared.db.database import Database
from shared.schema.investigation_schema import (
    ClarificationAnswer,
    EvidencePolicy,
    EvidenceRequirement,
    InvestigationConstraints,
    InvestigationRequest,
    RequestAnalysis,
    TaxonomyResolution,
)


def _write_csv(
    archive: zipfile.ZipFile,
    name: str,
    headers: list[str],
    rows: list[list[str]],
) -> None:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(rows)
    archive.writestr(name, stream.getvalue())


def _small_onet_archive(path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        _write_csv(
            archive,
            "db_30_3_csv/software_skills.csv",
            [
                "O*NET-SOC Code",
                "Title",
                "Workplace Example",
                "Element ID",
                "Element Name",
                "Hot Technology",
                "In Demand",
            ],
            [
                [
                    "15-1252.00",
                    "Software Developers",
                    "Python",
                    "2.E.1",
                    "Development",
                    "Y",
                    "Y",
                ]
            ],
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


class _FakeModel:
    def __init__(self, value):
        self.value = value

    def invoke(self, _messages):
        return self.value


class _RecordingFakeModel(_FakeModel):
    def __init__(self, value):
        super().__init__(value)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return self.value


def _unexpected_collection(*_args):
    raise AssertionError("수집 단계가 실행되면 안 됩니다.")


def test_onet_archive_tolerates_missing_occupation_file(tmp_path):
    db_path = tmp_path / "jobs.db"
    archive_path = tmp_path / "onet.zip"
    _small_onet_archive(archive_path)

    counts = import_onet_archive(db_path, archive_path)

    assert counts == {
        "occupations": 0,
        "occupation_aliases": 0,
        "skills": 1,
        "relations": 0,
    }


def test_local_domain_families_cover_each_soc_major_group_once():
    payload = json.loads(DEFAULT_LOCAL_SEED.read_text(encoding="utf-8"))
    assigned_groups = [
        str(group)
        for concept in payload["concepts"]
        for group in concept.get("soc_major_groups", [])
    ]

    assert set(assigned_groups) == {
        "11",
        "13",
        "15",
        "17",
        "19",
        "21",
        "23",
        "25",
        "27",
        "29",
        "31",
        "33",
        "35",
        "37",
        "39",
        "41",
        "43",
        "45",
        "47",
        "49",
        "51",
        "53",
        "55",
    }
    assert len(assigned_groups) == len(set(assigned_groups))


def test_common_short_job_terms_resolve_to_specific_local_occupations(tmp_path):
    taxonomy = _prepared_taxonomy(tmp_path / "jobs.db")

    assert taxonomy.resolve_occupation_concepts("AI 개발자") == [
        "l2c:occupation:ai_ml_engineer"
    ]
    assert taxonomy.resolve_occupation_concepts("iOS") == [
        "l2c:occupation:ios_engineer"
    ]


def test_search_taxonomy_constructor_does_not_initialize_storage(tmp_path):
    db_path = tmp_path / "jobs.db"

    SearchTaxonomyService(db_path)

    assert db_path.exists() is False


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
                required_sites=["wanted"],
            )
        ],
        InvestigationConstraints(),
        taxonomy_service=taxonomy,
    )
    connection = sqlite3.connect(db_path)
    try:
        status = connection.execute(
            "SELECT taxonomy_index_status FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()[0]
    finally:
        connection.close()

    assert report["search_ready_db_rows"] == 0
    assert status == "pending"


def test_onet_occupations_are_imported_under_local_domain_hierarchy(tmp_path):
    db_path = tmp_path / "jobs.db"
    archive_path = tmp_path / "onet.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        _write_csv(
            archive,
            "db_30_3_csv/occupation_data.csv",
            ["O*NET-SOC Code", "Title", "Description"],
            [["15-1252.00", "Software Developers", "Develop software systems."]],
        )
        _write_csv(
            archive,
            "db_30_3_csv/software_skills.csv",
            [
                "O*NET-SOC Code",
                "Title",
                "Workplace Example",
                "Element ID",
                "Element Name",
                "Hot Technology",
                "In Demand",
            ],
            [
                [
                    "15-1252.00",
                    "Software Developers",
                    "Python",
                    "2.E.1",
                    "Development",
                    "Y",
                    "Y",
                ]
            ],
        )

    counts = import_onet_archive(db_path, archive_path)
    taxonomy = _prepared_taxonomy(db_path)
    candidates = taxonomy.occupation_resolution_candidates(["l2c:domain:it_data"])

    assert counts == {
        "occupations": 1,
        "occupation_aliases": 1,
        "skills": 1,
        "relations": 0,
    }
    assert taxonomy.resolve_occupation_concepts("Software Developers") == [
        "onet:occupation:15-1252.00"
    ]
    assert "onet:occupation:15-1252.00" in {item["concept_key"] for item in candidates}


def test_generic_job_request_starts_with_all_occupation_domains(tmp_path):
    taxonomy = _prepared_taxonomy(tmp_path / "jobs.db")
    question = TaxonomyQuestionBuilder(taxonomy).build_next_scope_question(
        InvestigationConstraints(occupation_scope_required=True)
    )

    assert question is not None
    assert question.field == "occupation_domain_concept_keys"
    assert question.allow_custom is True
    assert {option.value for option in question.options} == {
        "l2c:domain:management_office_finance",
        "l2c:domain:it_data",
        "l2c:domain:research_engineering",
        "l2c:domain:production_manufacturing_maintenance",
        "l2c:domain:sales_service_logistics",
        "l2c:domain:healthcare_education_public",
    }
    assert all(option.concept_count > 0 for option in question.options)


def test_domain_selection_advances_to_family_options_even_without_jobs(tmp_path):
    taxonomy = _prepared_taxonomy(tmp_path / "jobs.db")
    domain_question = TaxonomyQuestionBuilder(taxonomy).build_next_scope_question(
        InvestigationConstraints(occupation_scope_required=True)
    )
    assert domain_question is not None
    it_option = next(
        option
        for option in domain_question.options
        if option.value == "l2c:domain:it_data"
    )
    investigation = InvestigationRequest(
        investigation_id="domain-choice",
        original_query="채용공고 찾아줘",
        constraints=InvestigationConstraints(occupation_scope_required=True),
        clarification_questions=[domain_question],
    )

    updated = apply_clarification_answer(
        investigation,
        ClarificationAnswer(
            question_id=domain_question.question_id,
            selected_option_id=it_option.option_id,
        ),
    )
    family_question = TaxonomyQuestionBuilder(taxonomy).build_next_scope_question(
        updated.constraints,
        answered_question_ids=[domain_question.question_id],
    )

    assert updated.constraints.occupation_domain_concept_keys == ["l2c:domain:it_data"]
    assert family_question is not None
    assert family_question.facet_type == "occupation_family"
    assert {option.value for option in family_question.options} == {
        "l2c:occupation:technology",
        "l2c:occupation:software_engineer",
        "l2c:occupation:data_engineer",
        "l2c:occupation:qa_engineer",
        "l2c:occupation:system_architect",
        "l2c:occupation:ai_ml_engineer",
    }
    assert all(option.matching_count == 0 for option in family_question.options)


def test_explicit_whole_domain_skips_family_question(tmp_path):
    taxonomy = _prepared_taxonomy(tmp_path / "jobs.db")
    constraints = taxonomy.enrich_constraints(
        InvestigationConstraints(
            occupation_domain_query="IT",
            occupation_scope_mode="all",
        )
    )

    assert constraints.occupation_domain_concept_keys == ["l2c:domain:it_data"]
    assert (
        TaxonomyQuestionBuilder(taxonomy).build_next_scope_question(constraints) is None
    )


def test_cardinality_question_lists_every_nonempty_direct_child(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    _insert_and_link(
        db, taxonomy, "backend-1", position="백엔드 개발자", job_category="백엔드 개발"
    )
    _insert_and_link(
        db,
        taxonomy,
        "backend-2",
        position="Backend Engineer",
        job_category="Backend Engineer",
    )
    _insert_and_link(
        db,
        taxonomy,
        "frontend",
        position="프론트엔드 개발자",
        job_category="프론트엔드 개발",
    )
    _insert_and_link(
        db, taxonomy, "ios", position="iOS 개발자", job_category="모바일 앱 개발"
    )
    _insert_and_link(
        db,
        taxonomy,
        "android",
        position="Android 개발자",
        job_category="Android 개발자",
    )

    constraints = taxonomy.enrich_constraints(
        InvestigationConstraints(occupation_query="소프트웨어 개발자")
    )
    question = TaxonomyQuestionBuilder(taxonomy).build_scope_question(constraints)

    assert constraints.occupation_concept_keys == ["l2c:occupation:software_engineer"]
    assert constraints.collection_search_term == "소프트웨어 개발자"
    assert question is not None
    counts = {
        option.collection_search_term: option.matching_count
        for option in question.options
        if option.collection_search_term
    }
    assert counts == {
        "백엔드 엔지니어": 2,
        "모바일 앱 개발자": 2,
        "프론트엔드 엔지니어": 1,
    }
    assert question.options[-1].matching_count == 5


def test_child_selection_updates_scope_and_collection_search_term(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    _insert_and_link(
        db, taxonomy, "ios", position="iOS 개발자", job_category="모바일 앱 개발"
    )
    _insert_and_link(
        db,
        taxonomy,
        "android",
        position="Android 개발자",
        job_category="Android 개발자",
    )
    constraints = taxonomy.enrich_constraints(
        InvestigationConstraints(occupation_query="모바일 개발자")
    )
    question = TaxonomyQuestionBuilder(taxonomy).build_scope_question(constraints)
    assert question is not None
    ios_option = next(
        option
        for option in question.options
        if option.collection_search_term == "iOS 개발자"
    )
    investigation = InvestigationRequest(
        investigation_id="taxonomy-choice",
        original_query="모바일 개발자 공고를 보여줘",
        constraints=constraints,
        clarification_questions=[question],
    )

    updated = apply_clarification_answer(
        investigation,
        ClarificationAnswer(
            question_id=question.question_id,
            selected_option_id=ios_option.option_id,
        ),
    )

    assert updated.constraints.occupation_concept_keys == [
        "l2c:occupation:ios_engineer"
    ]
    assert updated.constraints.occupation_query == "iOS 개발자"
    assert updated.constraints.collection_search_term == "iOS 개발자"
    assert updated.constraints.occupation_scope_mode == "selected"


def test_all_selection_preserves_original_collection_search_term(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    _insert_and_link(
        db, taxonomy, "backend", position="백엔드 개발자", job_category="백엔드 개발"
    )
    _insert_and_link(
        db,
        taxonomy,
        "frontend",
        position="프론트엔드 개발자",
        job_category="프론트엔드 개발",
    )
    _insert_and_link(db, taxonomy, "ai", position="AI 엔지니어", job_category="AI/ML")
    constraints = taxonomy.enrich_constraints(
        InvestigationConstraints(occupation_query="개발자")
    )
    question = TaxonomyQuestionBuilder(taxonomy).build_scope_question(constraints)
    assert question is not None
    investigation = InvestigationRequest(
        investigation_id="all-scope",
        original_query="개발자 공고를 보여줘",
        constraints=constraints,
        clarification_questions=[question],
    )

    updated = apply_clarification_answer(
        investigation,
        ClarificationAnswer(
            question_id=question.question_id,
            selected_option_id="all-descendants",
        ),
    )

    assert updated.constraints.occupation_scope_mode == "all"
    assert updated.constraints.collection_search_term == "개발자"


def test_dictionary_scope_does_not_union_unclassified_jobs(tmp_path):
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
        occupation_query="백엔드 개발자",
        occupation_concept_keys=["l2c:occupation:backend_engineer"],
    )

    report = inspect_job_evidence(db_path, [requirement], InvestigationConstraints())

    assert report["requirements"][0]["document_ids"] == [backend_id]
    assert report["requirements"][0]["semantic_review_required"] is False


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

    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        """
        SELECT concepts.concept_key, links.evidence_field, links.requirement_type
        FROM job_concept_links AS links
        JOIN search_concepts AS concepts ON concepts.id = links.concept_id
        WHERE links.job_id = ? AND links.link_type = 'skill'
        ORDER BY concepts.concept_key, links.evidence_field
        """,
        (job_id,),
    ).fetchall()
    connection.close()

    assert ("l2c:skill:swiftui", "requirements", "required") in rows
    assert ("l2c:skill:rxswift", "preferred", "preferred") in rows
    assert ("l2c:skill:flutter", "main_tasks", "mentioned") in rows
    swiftui_jobs = taxonomy.matching_skill_job_ids(
        ["l2c:skill:swiftui"],
        requirement_type="required",
    )
    assert swiftui_jobs == {job_id}


def test_candidate_can_be_accepted_as_existing_concept_alias(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    job_id = _insert_and_link(
        db,
        taxonomy,
        "unregistered-alias",
        position="iOS 개발자",
        tech_stack=["SwiftUI 5"],
    )
    review = SearchTaxonomyReviewService(db_path)
    candidate = review.list_candidates()[0]

    result = review.accept_as_alias(
        int(candidate["id"]),
        "l2c:skill:swiftui",
        note="버전이 붙은 동일 기술명",
    )

    assert result["status"] == "accepted"
    assert result["relinked"]["jobs"] == 1
    assert taxonomy.resolve_skill_concepts(["SwiftUI 5"]) == ["l2c:skill:swiftui"]
    assert taxonomy.matching_skill_job_ids(["l2c:skill:swiftui"]) == {job_id}
    assert Database(db_path).get(job_id)["tech_stack"] == ["SwiftUI 5"]


def test_candidate_can_be_accepted_as_new_curated_concept(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    job_id = _insert_and_link(
        db,
        taxonomy,
        "new-skill",
        position="JavaScript 개발자",
        tech_stack=["Bun Runtime"],
    )
    review = SearchTaxonomyReviewService(db_path)
    candidate = review.list_candidates()[0]

    result = review.accept_as_new_concept(
        int(candidate["id"]),
        "Bun",
        aliases=["Bun.js"],
    )

    concept_key = str(result["concept_key"])
    assert concept_key.startswith("l2c:curated:skill:")
    assert taxonomy.resolve_skill_concepts(["Bun Runtime"]) == [concept_key]
    assert taxonomy.resolve_skill_concepts(["Bun.js"]) == [concept_key]
    assert taxonomy.matching_skill_job_ids([concept_key]) == {job_id}


def test_reindex_resolves_candidate_that_now_exists_in_active_dictionary(tmp_path):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    job_id = _insert_and_link(
        db,
        taxonomy,
        "stale-candidate",
        position="iOS 개발자",
        tech_stack=["Future Swift"],
    )
    review = SearchTaxonomyReviewService(db_path)
    candidate = review.list_candidates()[0]
    review.accept_as_alias(
        int(candidate["id"]),
        "l2c:skill:swiftui",
    )

    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE search_term_candidates SET status = 'candidate', reviewed_at = NULL"
    )
    connection.commit()
    connection.close()

    totals = JobTaxonomyLinker(taxonomy.db_path).relink_all_jobs()
    accepted = review.list_candidates(status="accepted")

    assert totals["resolved_candidates"] == 1
    assert accepted[0]["accepted_concept_key"] == "l2c:skill:swiftui"
    assert taxonomy.matching_skill_job_ids(["l2c:skill:swiftui"]) == {job_id}


def test_reindex_is_idempotent_and_does_not_rewrite_source_tech_stack(tmp_path):
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
    connection = sqlite3.connect(db_path)
    first_links = connection.execute(
        "SELECT COUNT(*) FROM job_concept_links WHERE job_id = ?",
        (job_id,),
    ).fetchone()[0]
    connection.close()
    second = JobTaxonomyLinker(taxonomy.db_path).relink_all_jobs()
    connection = sqlite3.connect(db_path)
    second_links = connection.execute(
        "SELECT COUNT(*) FROM job_concept_links WHERE job_id = ?",
        (job_id,),
    ).fetchone()[0]
    connection.close()

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

    assert (
        taxonomy.matching_occupation_job_ids(["l2c:occupation:ai_ml_engineer"]) == set()
    )

    retry = JobTaxonomyLinker(taxonomy.db_path).relink_pending_jobs(
        limit=10, max_attempts=2
    )

    with sqlite3.connect(db_path) as connection:
        status = connection.execute(
            """
            SELECT taxonomy_index_status, taxonomy_index_attempts,
                   taxonomy_index_error
            FROM jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
    assert retry["indexed"] == 1
    assert status == ("indexed", 2, None)
    assert taxonomy.matching_occupation_job_ids(["l2c:occupation:ai_ml_engineer"]) == {
        job_id
    }
    assert (
        JobTaxonomyLinker(taxonomy.db_path).relink_pending_jobs(
            limit=10, max_attempts=2
        )["jobs"]
        == 0
    )


def test_onet_reimport_preserves_curated_alias_on_existing_concept(tmp_path):
    db_path = tmp_path / "jobs.db"
    archive_path = tmp_path / "onet.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        _write_csv(
            archive,
            "db_30_3_csv/software_skills.csv",
            [
                "O*NET-SOC Code",
                "Title",
                "Workplace Example",
                "Element ID",
                "Element Name",
                "Hot Technology",
                "In Demand",
            ],
            [
                [
                    "15-1252.00",
                    "Software Developers",
                    "External Runtime",
                    "2.E.1",
                    "Development",
                    "Y",
                    "Y",
                ]
            ],
        )
    import_onet_archive(db_path, archive_path)
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    job_id = _insert_and_link(
        db,
        taxonomy,
        "external-alias",
        position="데이터 개발자",
        tech_stack=["ER Tool"],
    )
    connection = sqlite3.connect(db_path)
    onet_key = connection.execute(
        """
        SELECT concept_key
        FROM search_concepts
        WHERE source_key = 'onet_30_3' AND preferred_label_en = 'External Runtime'
        """
    ).fetchone()[0]
    connection.close()
    review = SearchTaxonomyReviewService(db_path)
    candidate = review.list_candidates()[0]
    review.accept_as_alias(int(candidate["id"]), str(onet_key))

    import_onet_archive(db_path, archive_path)
    _prepared_taxonomy(db_path)
    JobTaxonomyLinker(db_path).relink_all_jobs()

    assert taxonomy.resolve_skill_concepts(["ER Tool"]) == [onet_key]
    assert taxonomy.matching_skill_job_ids([onet_key]) == {job_id}


def test_workflow_returns_cardinality_question_before_evidence_planning(
    tmp_path,
    investigation_workflow_factory,
):
    db_path = tmp_path / "jobs.db"
    db = Database(db_path)
    taxonomy = _prepared_taxonomy(db_path)
    _insert_and_link(
        db, taxonomy, "backend", position="백엔드 개발자", job_category="백엔드 개발"
    )
    _insert_and_link(
        db,
        taxonomy,
        "frontend",
        position="프론트엔드 개발자",
        job_category="프론트엔드 개발",
    )
    workflow = investigation_workflow_factory(
        db_path=db_path,
        run_collection=_unexpected_collection,
        postprocess_collection=_unexpected_collection,
        store_collection=_unexpected_collection,
        record_experience=_unexpected_collection,
        taxonomy_service=taxonomy,
        models=InvestigationModels(
            analysis_model=_FakeModel(
                RequestAnalysis(
                    objective="소프트웨어 개발 공고 조회",
                    deliverable="공고 목록",
                    constraints=InvestigationConstraints(
                        occupation_query="소프트웨어 개발자"
                    ),
                )
            )
        ),
    )

    result = workflow.run("소프트웨어 개발자 공고를 보여줘")

    assert result.run_status.value == "waiting_input"
    assert result.clarification["field"] == "occupation_concept_keys"
    assert result.clarification["candidate_count"] == 2
    assert [option["matching_count"] for option in result.clarification["options"]] == [
        1,
        1,
        2,
    ]


def test_workflow_progresses_from_generic_request_to_domain_then_family(
    tmp_path,
    investigation_workflow_factory,
):
    db_path = tmp_path / "jobs.db"
    taxonomy = _prepared_taxonomy(db_path)
    workflow = investigation_workflow_factory(
        db_path=db_path,
        run_collection=_unexpected_collection,
        postprocess_collection=_unexpected_collection,
        store_collection=_unexpected_collection,
        record_experience=_unexpected_collection,
        taxonomy_service=taxonomy,
        models=InvestigationModels(
            analysis_model=_FakeModel(
                RequestAnalysis(
                    objective="채용공고 찾기",
                    deliverable="공고 목록",
                    constraints=InvestigationConstraints(
                        occupation_scope_required=True,
                    ),
                )
            )
        ),
    )

    first = workflow.run("채용공고 찾아줘")
    it_option = next(
        option
        for option in first.clarification["options"]
        if option["value"] == "l2c:domain:it_data"
    )
    second = workflow.run(
        "",
        investigation_id=first.investigation.investigation_id,
        clarification_answer=ClarificationAnswer(
            question_id=first.clarification["question_id"],
            selected_option_id=it_option["option_id"],
        ),
    )

    assert first.clarification["facet_type"] == "occupation_domain"
    assert second.run_status.value == "waiting_input"
    assert second.clarification["facet_type"] == "occupation_family"
    assert "l2c:occupation:technology" in {
        option["value"] for option in second.clarification["options"]
    }


def test_semantic_resolution_uses_selected_domain_and_promotes_confirmed_alias(
    tmp_path,
    investigation_workflow_factory,
):
    db_path = tmp_path / "jobs.db"
    taxonomy = _prepared_taxonomy(db_path)
    taxonomy_model = _RecordingFakeModel(
        TaxonomyResolution(
            decision="match",
            selected_concept_key="l2c:occupation:llm_engineer",
            reason="LLM 응용 개발 역할로 해석",
        )
    )
    workflow = investigation_workflow_factory(
        db_path=db_path,
        run_collection=_unexpected_collection,
        postprocess_collection=_unexpected_collection,
        store_collection=_unexpected_collection,
        record_experience=_unexpected_collection,
        taxonomy_service=taxonomy,
        models=InvestigationModels(
            analysis_model=_FakeModel(
                RequestAnalysis(
                    objective="프롬프트 엔지니어 설명",
                    deliverable="직무 설명",
                    evidence_policy=EvidencePolicy.MODEL_KNOWLEDGE,
                    constraints=InvestigationConstraints(
                        occupation_domain_query="IT",
                        occupation_query="프롬프트 엔지니어",
                    ),
                )
            ),
            taxonomy_model=taxonomy_model,
            answer_model=_FakeModel("프롬프트 엔지니어 직무 설명"),
        ),
    )

    first = workflow.run("프롬프트 엔지니어는 무슨 일을 해?")

    assert first.run_status.value == "waiting_input"
    assert first.clarification["facet_type"] == "semantic_occupation"
    assert [option["value"] for option in first.clarification["options"]] == [
        "l2c:occupation:llm_engineer"
    ]
    model_payload = taxonomy_model.messages[0][1].content
    assert "l2c:occupation:llm_engineer" in model_payload
    assert "l2c:occupation:legal" not in model_payload

    option = first.clarification["options"][0]
    second = workflow.run(
        "",
        investigation_id=first.investigation.investigation_id,
        clarification_answer=ClarificationAnswer(
            question_id=first.clarification["question_id"],
            selected_option_id=option["option_id"],
        ),
    )

    assert second.final_answer == "프롬프트 엔지니어 직무 설명"
    assert taxonomy.resolve_occupation_concepts("프롬프트 엔지니어") == [
        "l2c:occupation:llm_engineer"
    ]
    accepted = SearchTaxonomyReviewService(db_path).list_candidates(status="accepted")
    assert accepted[0]["accepted_concept_key"] == "l2c:occupation:llm_engineer"
