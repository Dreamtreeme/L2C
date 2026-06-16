import json
import logging
import os
from typing import Any

from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

logger = logging.getLogger(__name__)

DEFAULT_RECURSION_LIMIT = 60


def _default_site_slug() -> str:
    from agent.sites import list_supported_sites

    sites = list_supported_sites()
    if not sites:
        raise ValueError("No enabled site profiles are configured")
    return sites[0]["slug"]


def _load_collection_profile(site: str | None) -> dict:
    from agent.sites import load_site_profile

    return load_site_profile(site or _default_site_slug())


def _join_manual_items(items) -> str:
    if not isinstance(items, list):
        return ""
    return "; ".join(str(item) for item in items if item)


def _build_site_goal(search_keyword: str, profile: dict) -> str:
    entry = profile["entry"]
    manual = profile["manual"]
    tools = profile["tools"]
    site_name = entry.get("display_name") or entry.get("slug")
    base_url = entry.get("base_url") or manual.get("base_url")
    common_flow = _join_manual_items(manual.get("common_flow", []))
    stable_controls = _join_manual_items(manual.get("stable_controls", []))
    variable_entities = _join_manual_items(manual.get("variable_entities", []))
    ignore_elements = _join_manual_items(manual.get("ignore_elements", []))
    required_fields = _join_manual_items(manual.get("collection_policy", {}).get("required_fields", []))
    safe_reflex = _join_manual_items(manual.get("reflex_policy", {}).get("safe_actions", []))
    unsafe_reflex = _join_manual_items(manual.get("reflex_policy", {}).get("unsafe_actions", []))
    allowed_tools = _join_manual_items(tools.get("allowed_tools", []))
    site_prompt = profile.get("prompt", "").strip()

    return (
        f"{site_name}({base_url})에서 '{search_keyword}' 채용공고를 검색하고 수집하세요. "
        "검색 결과 화면에 도달하면 공고 수와 카드/행 목록을 확인하고, "
        "방문할 공고별 순회 계획을 세운 뒤 상세 페이지를 하나씩 방문하세요. "
        "상세 페이지에서 필요한 정보를 수집한 뒤 목록으로 돌아와 이미 클릭한 공고는 제외하고 다음 공고를 선택하세요.\n\n"
        f"[사이트 공통 흐름]\n{common_flow}\n\n"
        f"[안정적인 UI/Reflex 후보]\n{stable_controls}\n\n"
        f"[목표나 실행 시점에 따라 달라지는 UI]\n{variable_entities}\n\n"
        f"[무시할 요소]\n{ignore_elements}\n\n"
        f"[필수 수집 필드]\n{required_fields}\n\n"
        f"[Reflex 안전 액션]\n{safe_reflex}\n\n"
        f"[Reflex 금지/주의 액션]\n{unsafe_reflex}\n\n"
        f"[허용 도구]\n{allowed_tools}\n\n"
        f"[하위 에이전트 사이트 지침]\n{site_prompt}"
    )


def _run_graph_with_last_state(app: Any, initial_state: dict, recursion_limit: int) -> tuple[dict, bool]:
    """
    LangGraph 실행 중 recursion limit에 걸려도 마지막 state를 보존합니다.
    app.invoke()는 예외 시 partial state를 돌려주지 않으므로 stream(values)을 사용합니다.
    """
    last_state = initial_state
    try:
        for state in app.stream(
            initial_state,
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            if isinstance(state, dict):
                last_state = state
        return last_state, False
    except GraphRecursionError as e:
        logger.warning(f"[realtime_scraping] Graph recursion limit reached; preserving partial state: {e}")
        return last_state, True




def _close_browser_after_run() -> None:
    if os.getenv("VISION_CLOSE_BROWSER_AFTER_RUN", "1").strip().lower() in {"0", "false", "no", "off"}:
        logger.info("[realtime_scraping] Browser cleanup disabled")
        return
    try:
        from agent.graph import nodes

        action_tools = getattr(nodes, "_action_tools", None)
        if action_tools is None:
            logger.info("[realtime_scraping] Browser cleanup skipped: action tools were not initialized")
            return
        result = action_tools.close_browser()
        logger.info("[realtime_scraping] Browser cleanup completed: %s", result)
    except Exception as e:
        logger.debug("[realtime_scraping] Browser cleanup skipped: %s", e)

def _commit_feedback_episodes(final_state: dict, hit_recursion_limit: bool, is_finished: bool) -> int:
    """Persist feedback episodes for later Critic/Recipe Memory promotion. Best-effort."""
    episodes = list(final_state.get("feedback_episodes", []) or [])
    if not episodes:
        return 0
    run_status = "finished" if is_finished else "recursion_limit" if hit_recursion_limit else "stopped"
    try:
        from agent.recipe.feedback_store import FeedbackStore

        saved = FeedbackStore().commit_episodes(
            episodes,
            run_status=run_status,
            source="realtime_scraping",
        )
        logger.info(
            "[realtime_scraping] Feedback episodes committed: episodes=%s, saved=%s, status=%s",
            len(episodes),
            saved,
            run_status,
        )
        return saved
    except Exception as e:
        logger.debug(f"[realtime_scraping] Feedback episode commit skipped: {e}")
        return 0



def _worker_run_status(hit_recursion_limit: bool, is_finished: bool) -> str:
    if is_finished:
        return "finished"
    if hit_recursion_limit:
        return "recursion_limit"
    return "stopped"


def _worker_review_retries() -> int:
    try:
        return max(0, int(os.getenv("VISION_WORKER_REVIEW_RETRIES", "1")))
    except ValueError:
        return 1


def _append_review_feedback(goal: str, review_feedback: str | None) -> str:
    if not review_feedback:
        return goal
    return (
        goal
        + "\n\n[Commander review feedback from the previous attempt]\n"
        + review_feedback.strip()
        + "\nRevise the next actions and final submission to address this feedback."
    )


def _initial_worker_state(goal: str) -> dict:
    return {
        "goal": goal,
        "ui_context": "",
        "current_url": "",
        "current_url_stale": True,
        "current_markers": [],
        "action_history": [],
        "recent_images": [],
        "marked_image": "",
        "error_count": 0,
        "is_finished": False,
        "collected_data": [],
        "extracted_jd": {},
        "last_action_result": None,
        "plan": [],
        "current_plan_step": 0,
        "step_durations": [],
        "last_action_screen_changed": True,
        "recorded_steps": [],
        "feedback_episodes": [],
        "reflex_state_key": "",
        "reflex_hit": False,
        "reflex_expected_next_state": "",
        "reflex_pending_validation": False,
        "ocr_texts": [],
        "ocr_delta_added": [],
        "ocr_delta_removed": [],
        "reflex_validation_status": "",
    }


def run_worker_once(
    search_keyword: str,
    site: str | None = None,
    review_feedback: str | None = None,
    review_attempt: int = 0,
    run_id: str | None = None,
) -> dict:
    """Run one child vision worker attempt and return an unreviewed submission payload."""
    from agent.graph.workflow import build_graph
    from agent.recipe.reviewer import build_worker_submission

    app = build_graph()
    site_profile = _load_collection_profile(site)
    site_entry = site_profile["entry"]
    site_slug = site_entry.get("slug") or site or "unknown"
    site_name = site_entry.get("display_name") or site_slug
    goal = _append_review_feedback(_build_site_goal(search_keyword, site_profile), review_feedback)
    initial_state = _initial_worker_state(goal)

    logger.info(
        "[realtime_scraping] Starting worker graph site=%s attempt=%s",
        site_slug,
        review_attempt,
    )
    recursion_limit = int(os.getenv("VISION_AGENT_RECURSION_LIMIT", str(DEFAULT_RECURSION_LIMIT)))
    final_state, hit_recursion_limit = _run_graph_with_last_state(app, initial_state, recursion_limit)

    extracted = final_state.get("extracted_jd", {}) or {}
    is_finished = bool(final_state.get("is_finished", False))
    run_status = _worker_run_status(hit_recursion_limit, is_finished)
    feedback_saved = _commit_feedback_episodes(final_state, hit_recursion_limit, is_finished)
    submission = build_worker_submission(
        final_state,
        site=site_slug,
        keyword=search_keyword,
        run_status=run_status,
        hit_recursion_limit=hit_recursion_limit,
        persisted_count=0,
        feedback_saved=feedback_saved,
        review_attempt=review_attempt,
        run_id=run_id,
    )

    return {
        "submission": submission,
        "extracted_jd": extracted,
        "final_state": final_state,
        "site_slug": site_slug,
        "site_name": site_name,
        "keyword": search_keyword,
        "run_status": run_status,
        "hit_recursion_limit": hit_recursion_limit,
        "is_finished": is_finished,
        "feedback_saved": feedback_saved,
    }


def commit_worker_review(
    submission: dict,
    source: str = "realtime_scraping",
) -> tuple[dict, str]:
    """Review and persist one worker submission."""
    from agent.recipe.reviewer import review_worker_submission
    from agent.recipe.submission_store import SubmissionStore

    review = review_worker_submission(submission)
    submission_id = SubmissionStore().commit_submission(
        submission,
        review=review,
        source=source,
    )
    logger.info(
        "[realtime_scraping] Worker submission reviewed: id=%s decision=%s confidence=%s",
        submission_id,
        review.get("decision"),
        review.get("confidence"),
    )
    return review, submission_id


def persist_accepted_worker_result(worker_result: dict, review: dict, source: str = "realtime_scraping") -> tuple[int, dict, dict, str]:
    """Persist accepted worker data and update the stored submission row."""
    submission = dict(worker_result.get("submission") or {})
    if review.get("decision") != "accept" or not worker_result.get("extracted_jd"):
        return 0, submission, review, ""

    persisted_count = _persist_collected_data(worker_result.get("extracted_jd") or {}, worker_result.get("keyword", ""))
    submission["persisted_count"] = persisted_count
    worker_result["submission"] = submission
    from agent.recipe.submission_store import SubmissionStore

    submission_id = SubmissionStore().commit_submission(
        submission,
        review=review,
        source=source,
    )
    return persisted_count, submission, review, submission_id


def _result_payload(
    *,
    message: str,
    site_name: str,
    site_slug: str,
    keyword: str,
    item_count: int,
    persisted_count: int,
    submission_id: str,
    review: dict,
    hit_recursion_limit: bool,
    is_finished: bool,
) -> str:
    return json.dumps(
        {
            "message": message,
            "site": site_slug,
            "site_name": site_name,
            "keyword": keyword,
            "item_count": item_count,
            "persisted_count": persisted_count,
            "submission_id": submission_id,
            "review": review,
            "hit_recursion_limit": hit_recursion_limit,
            "is_finished": is_finished,
        },
        ensure_ascii=False,
        indent=2,
    )


@tool
def realtime_scraping(
    company: str = None,
    tech_stack: str = None,
    site: str = None,
    query: str = None,
    review_feedback: str = None,
    review_attempt: int = 0,
) -> str:
    """
    Run the child vision worker, review its structured submission, and persist only accepted data.
    """
    search_keyword = ""
    if query:
        search_keyword = query
    elif company and tech_stack:
        search_keyword = f"{company} {tech_stack}"
    elif company:
        search_keyword = company
    elif tech_stack:
        search_keyword = tech_stack
    else:
        return json.dumps({"message": "collection failed: missing search keyword", "review": {"decision": "reject"}}, ensure_ascii=False)

    logger.info("[realtime_scraping] Invoking vision worker for keyword=%r", search_keyword)

    try:
        from agent.recipe.reviewer import render_review_feedback

        max_review_retries = _worker_review_retries()
        attempt = max(0, int(review_attempt or 0))
        pending_feedback = review_feedback or ""
        run_id = None
        while True:
            worker_result = run_worker_once(
                search_keyword,
                site=site,
                review_feedback=pending_feedback,
                review_attempt=attempt,
                run_id=run_id,
            )
            submission = worker_result["submission"]
            run_id = submission.get("run_id") or run_id
            review, submission_id = commit_worker_review(submission)

            if review.get("decision") == "revise" and attempt < max_review_retries:
                pending_feedback = render_review_feedback(review)
                attempt += 1
                logger.info("[realtime_scraping] Retrying worker after commander feedback: attempt=%s", attempt)
                continue

            persisted_count, submission, review, persisted_submission_id = persist_accepted_worker_result(worker_result, review)
            if persisted_submission_id:
                submission_id = persisted_submission_id

            item_count = int(submission.get("collected_count") or 0)
            site_name = worker_result.get("site_name", site or "unknown")
            site_slug = worker_result.get("site_slug", site or "unknown")
            hit_recursion_limit = bool(worker_result.get("hit_recursion_limit", False))
            is_finished = bool(worker_result.get("is_finished", False))

            if review.get("decision") == "accept" and persisted_count > 0:
                completion_type = "partial collection persisted" if hit_recursion_limit and not is_finished else "vision collection persisted"
                message = f"{completion_type}: keyword={search_keyword!r}, site={site_name}, collected={item_count}, persisted={persisted_count}"
            elif review.get("decision") == "revise":
                feedback_text = review.get("feedback_to_worker") or "; ".join(review.get("reasons") or [])
                message = f"worker submission needs revision: {feedback_text}"
            elif hit_recursion_limit:
                message = f"collection stopped at recursion limit without accepted data: site={site_name}, keyword={search_keyword!r}"
            else:
                message = f"collection finished without accepted data: site={site_name}, keyword={search_keyword!r}"

            return _result_payload(
                message=message,
                site_name=site_name,
                site_slug=site_slug,
                keyword=search_keyword,
                item_count=item_count,
                persisted_count=persisted_count,
                submission_id=submission_id,
                review=review,
                hit_recursion_limit=hit_recursion_limit,
                is_finished=is_finished,
            )

    except Exception as e:
        logger.error("[realtime_scraping] Vision worker execution error: %s", e, exc_info=True)
        return json.dumps({"message": f"collection error: {e}", "review": {"decision": "reject"}}, ensure_ascii=False)
    finally:
        _close_browser_after_run()


def _persist_collected_data(extracted_jd: dict, keyword: str) -> int:
    """
    비전 에이전트가 수집한 extracted_jd 데이터를 전처리 후 DB에 UPSERT합니다.
    extracted_jd는 단일 공고 dict이거나, '공고목록' 키 아래 리스트를 담고 있을 수 있습니다.
    전처리(clean_text / extract_tech_stacks / parse_experience / generate_content_hash)는
    Preprocessor.process_raw_jd 에 전임합니다.
    """
    from shared.config import DB_PATH
    from shared.db.database import Database
    from agent.utils.preprocessor import Preprocessor

    db = Database(DB_PATH)

    # 공고 목록 추출 (리스트 or 단건)
    if "공고목록" in extracted_jd:
        job_list = extracted_jd["공고목록"]
        if not isinstance(job_list, list):
            job_list = [job_list]
    else:
        job_list = [extracted_jd] if extracted_jd else []

    persisted_count = 0
    for idx, job in enumerate(job_list):
        if not isinstance(job, dict) or not job:
            continue

        # fallback으로 검색 결과 페이지 URL을 저장하면 이후 원본 공고 조회 시 엉뚱한 페이지로 연결됩니다.
        # URL을 수집하지 못한 경우 해당 공고는 적재를 건너뜁니다.
        url = job.get("url") or job.get("URL")
        if not url:
            company_name = job.get("회사명", job.get("company_name", ""))
            position = job.get("직무명", job.get("position", ""))
            logger.warning(f"[_persist] Skipping job #{idx} ({company_name} - {position}): URL not collected")
            continue

        try:
            job.setdefault("url", url)
            job_posting = Preprocessor.process_raw_jd(job)
            db.upsert(url=url, data=job_posting.model_dump())
            persisted_count += 1
            logger.info(f"[_persist] Successfully upserted job #{idx}: {job_posting.company_name} - {job_posting.position}")

        except Exception as e:
            logger.error(f"[_persist] Failed to persist job #{idx}: {e}")
            continue
    return persisted_count
