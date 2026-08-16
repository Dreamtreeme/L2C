"""상세 OCR 초안과 작업자 검토 상태 전이를 검증한다."""

from agent.graph import worker_execution_dispatch
from agent.graph.worker_execution_policy import compact_action_args
from agent.graph.worker_review import review_node
from agent.runtime.detail_runtime import update_job_detail_buffer
from agent.runtime.tool_schema import scroll
from agent.tests.worker_test_support import node_runtime, worker_data_services, worker_state
from shared.schema.collection_intent import CollectionIntent
from shared.schema.jd_schema import JobPosting, JobReview, JobReviewStatus


REQUIRED_FIELDS = [
    "company_name",
    "position",
    "url",
    "main_tasks",
    "requirements",
]


def _detail_state(current_url: str):
    detail_buffer = update_job_detail_buffer(
        None,
        [
            {"id": 3, "text": "예시회사", "bbox": [50, 100, 150, 130], "type": "text"},
            {"id": 5, "text": "AI 엔지니어", "bbox": [170, 100, 400, 130], "type": "text"},
            {"id": 20, "text": "주요 업무 모델 운영", "bbox": [50, 400, 500, 430], "type": "text"},
            {"id": 30, "text": "자격 요건 Python", "bbox": [50, 600, 500, 630], "type": "text"},
        ],
        current_url,
        "detail.png",
        page_role="job_detail",
        detail_key="card-1",
        screen_size=[1000, 1000],
    )
    return worker_state(
        request={
            "collection_intent": CollectionIntent(required_fields=REQUIRED_FIELDS)
        },
        observation={"current_url": current_url, "current_page_role": "job_detail"},
        transition={
            "transition_result": {
                "status": "unknown",
                "action": "scroll",
                "reason": "no_screen_change",
                "needs_ocr": False,
            }
        },
        collection={
            "job_card_queue": [
                {
                    "queue_id": "card-1",
                    "status": "active",
                    "company": "예시회사",
                    "title": "AI 엔지니어",
                }
            ],
            "job_detail_buffer": detail_buffer,
        },
    )


def _request_review(state):
    outcome = worker_execution_dispatch.dispatch_state_action(
        "review_job_detail",
        {"reason": "누적 근거를 검토합니다."},
        current_url=state["observation"]["current_url"],
        state=state,
        data_services=worker_data_services(),
    )
    return outcome


def test_physical_actions_do_not_accept_field_self_reports():
    assert "observed_fields" not in scroll.model_fields


def test_review_request_creates_draft_without_completing_job():
    current_url = "https://www.wanted.co.kr/wd/1"
    state = _detail_state(current_url)

    outcome = _request_review(state)

    assert outcome.result["status"] == "success"
    update = outcome.state_update["collection"]
    draft = update["pending_job_draft"]
    assert draft.url == current_url
    assert draft.detail_key == "card-1"
    assert draft.screen_count == 1
    assert "자격 요건 Python" in draft.raw_ocr_text
    assert draft.target_company_name == "예시회사"
    assert draft.target_position == "AI 엔지니어"
    assert draft.ocr_items[0].id == 1
    assert draft.ocr_items[0].bbox_ratio == [0.05, 0.1, 0.4, 0.13]
    assert "job_captures" not in update
    assert "job_detail_buffer" not in update


def test_complete_review_is_the_only_path_that_counts_job():
    state = _detail_state("https://www.wanted.co.kr/wd/1")
    draft = _request_review(state).state_update["collection"]["pending_job_draft"]
    state["collection"]["pending_job_draft"] = draft
    review = JobReview(
        detail_key=draft.detail_key,
        url=draft.url,
        status=JobReviewStatus.COMPLETE,
        posting=JobPosting(
            company_name="예시회사",
            position="AI 엔지니어",
            url=draft.url,
            main_tasks=["모델 운영"],
            requirements=["Python"],
        ),
        field_evidence={
            "company_name": "예시회사",
            "position": "AI 엔지니어",
            "url": draft.url,
            "main_tasks": "주요 업무 모델 운영",
            "requirements": "자격 요건 Python",
        },
    )

    update = review_node(
        state,
        node_runtime(
            data=worker_data_services(review_job_draft=lambda _draft, _intent: review)
        ),
    )

    collection = update["collection"]
    assert len(collection["job_captures"]) == 1
    assert len(collection["collected_jobs"]) == 1
    assert collection["job_card_queue"][0]["status"] == "done"
    assert collection["job_detail_buffer"] == {}


def test_needs_more_review_keeps_detail_buffer_and_card_active():
    state = _detail_state("https://www.wanted.co.kr/wd/1")
    draft = _request_review(state).state_update["collection"]["pending_job_draft"]
    state["collection"]["pending_job_draft"] = draft
    review = JobReview(
        detail_key=draft.detail_key,
        url=draft.url,
        status=JobReviewStatus.NEEDS_MORE,
        missing_fields=["requirements"],
        draft_fingerprint=draft.fingerprint(),
        reason="자격 요건 본문이 더 필요합니다.",
    )

    update = review_node(
        state,
        node_runtime(
            data=worker_data_services(review_job_draft=lambda _draft, _intent: review)
        ),
    )

    collection = update["collection"]
    assert collection["pending_job_draft"] is None
    assert "job_detail_buffer" not in collection
    assert "job_captures" not in collection
    assert state["collection"]["job_card_queue"][0]["status"] == "active"

    state["collection"].update(collection)
    repeated = _request_review(state)
    assert repeated.result["status"] == "success"
    assert repeated.state_update["collection"][
        "pending_job_draft"
    ].review_model_tier == "primary"

    state["collection"]["last_job_review"] = review.model_copy(
        update={"model_tier": "primary"}
    )
    repeated_primary = _request_review(state)
    assert repeated_primary.result["status"] == "skipped"
    assert repeated_primary.result["reason"] == "detail_evidence_unchanged"


def test_source_incomplete_review_rejects_card_without_counting_job():
    state = _detail_state("https://www.wanted.co.kr/wd/1")
    draft = _request_review(state).state_update["collection"]["pending_job_draft"]
    state["collection"]["pending_job_draft"] = draft
    review = JobReview(
        detail_key=draft.detail_key,
        url=draft.url,
        status=JobReviewStatus.SOURCE_INCOMPLETE,
        missing_fields=["requirements"],
        reason="공고 원문에 자격 요건이 없습니다.",
    )

    update = review_node(
        state,
        node_runtime(
            data=worker_data_services(review_job_draft=lambda _draft, _intent: review)
        ),
    )

    collection = update["collection"]
    assert collection["job_card_queue"][0]["status"] == "rejected"
    assert "job_captures" not in collection
    assert collection["job_detail_buffer"] == {}


def test_review_action_args_compaction_is_idempotent():
    args = {"page_role": "job_detail", "reason": "근거 검토"}

    compacted = compact_action_args("review_job_detail", args)

    assert compact_action_args("review_job_detail", compacted) == compacted
