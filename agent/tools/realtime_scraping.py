import json
import logging
import os
from typing import Any
from urllib.parse import quote_plus

from langchain_core.tools import tool
from agent.application.job_persistence_service import (
    normalize_job_for_persistence as _normalize_job_for_persistence,
    persist_collected_data as _persist_collected_data,
)
from agent.application.worker_execution_service import (
    close_browser_after_run as _close_browser_after_run,
    execute_worker_graph as _execute_worker_graph,
    prepare_worker_start_screen as _prepare_worker_start_screen,
    run_graph_with_last_state as _run_graph_with_last_state,
    worker_execution_session,
    worker_preopen_enabled as _worker_preopen_enabled,
    worker_start_url as _worker_start_url,
)
from agent.recipe.task_category import DEFAULT_SEARCH_TASK_CATEGORY, normalize_task_category
from agent.runtime.job_collection import job_list_value as _job_list_value
from agent.utils.model_dump import dump_model
from shared.schema.collection_intent import (
    CollectionCountMode,
    CollectionIntent,
    normalize_collection_intent,
)

logger = logging.getLogger(__name__)

DEFAULT_RECURSION_LIMIT = 60


SearchIntent = CollectionIntent


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _normalize_target_count(value: Any) -> int:
    """사용자가 요청한 수집 개수를 안전한 정수 범위로 정규화한다."""
    try:
        count = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, count))


def _suggested_recursion_limit(current_limit: int) -> int:
    increment = max(1, _env_int("VISION_AGENT_RECURSION_LIMIT_INCREMENT", current_limit))
    return current_limit + increment


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


def _profile_site_terms(profile: dict) -> list[str]:
    entry = profile.get("entry", {}) if isinstance(profile, dict) else {}
    manual = profile.get("manual", {}) if isinstance(profile, dict) else {}
    terms = [
        entry.get("slug"),
        entry.get("display_name"),
        manual.get("site"),
        manual.get("display_name"),
    ]
    terms.extend(entry.get("domains", []) or [])
    base_url = entry.get("base_url") or manual.get("base_url")
    if base_url:
        terms.append(str(base_url).replace("https://", "").replace("http://", "").strip("/"))
    cleaned = {str(term).strip() for term in terms if str(term or "").strip()}
    return sorted(cleaned, key=len, reverse=True)


def _search_intent_mode() -> str:
    mode = os.getenv("VISION_SEARCH_INTENT_MODE", "llm").strip().lower()
    return mode if mode in {"llm", "off"} else "llm"


def _extract_search_intent(raw_query: str, profile: dict) -> dict[str, Any]:
    """사용자 요청에서 작업자에게 전달할 전체 수집 조건을 추출한다."""
    original = str(raw_query or "").strip()
    if not original:
        return dump_model(CollectionIntent())

    mode = _search_intent_mode()
    if mode == "off":
        intent = dump_model(CollectionIntent(original_query=original, search_keyword=original))
        intent["source"] = "disabled"
        intent["error"] = ""
        return intent

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from agent.application.model_clients import get_structured_google_model
        from shared.config import BASE_DIR  # noqa: F401 - .env 로드를 보장한다.

        entry = profile.get("entry", {}) if isinstance(profile, dict) else {}
        manual = profile.get("manual", {}) if isinstance(profile, dict) else {}
        model_name = os.getenv("VISION_SEARCH_INTENT_MODEL", os.getenv("VISION_WORKER_REVIEW_MODEL", "gemini-3.5-flash"))
        llm = get_structured_google_model(model_name, CollectionIntent, temperature=0.0)
        messages = [
            SystemMessage(
                content=(
                    "Extract a job-search intent for an autonomous vision worker. "
                    "Return only the actual phrase that should be searched on the target job site. "
                    "Remove site names, URLs, filler commands, and analysis/reporting words. "
                    "Preserve date, experience, location, employment type, freshness, and analysis purpose. "
                    "Use count_mode=explicit only for an explicit number and set target_count. "
                    "Use count_mode=visible_all for all/every posting or when no count is specified. "
                    "Use purpose=compare or trend only when requested. "
                    "Do not translate or broaden the keyword."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "user_request": original,
                        "site": entry.get("slug") or manual.get("site"),
                        "site_terms": _profile_site_terms(profile),
                        "default_query_target": manual.get("default_query_target", ""),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        ]
        from agent.application.run_context import invoke_with_metrics

        data = dump_model(invoke_with_metrics(llm, messages, "search_intent"))
        entry_site = str(entry.get("slug") or manual.get("site") or "")
        intent = dump_model(
            normalize_collection_intent(
                data,
                original_query=original,
                site=entry_site,
                search_keyword=original,
            )
        )
        intent["source"] = "llm"
        intent["error"] = ""
        return intent
    except Exception as exc:  # pragma: no cover - provider failures are best-effort
        logger.warning("[realtime_scraping] Search intent extraction failed; using raw query: %s", exc)
        intent = dump_model(CollectionIntent(original_query=original, search_keyword=original))
        intent["source"] = "llm_failed"
        intent["error"] = str(exc)[:200]
        return intent


def _build_direct_search_url(search_keyword: str, profile: dict) -> str:
    manual = profile.get("manual", {}) if isinstance(profile, dict) else {}
    navigation = manual.get("navigation_policy", {}) if isinstance(manual, dict) else {}
    if navigation.get("allow_direct_search_url") is False:
        return ""
    template = str(navigation.get("search_url_template") or "").strip()
    if not search_keyword or not template:
        return ""
    encoded = quote_plus(search_keyword)
    return template.format(query=encoded, keyword=encoded, raw_query=search_keyword)


def _compact_research_for_goal(research: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(research, dict):
        return {}
    out: dict[str, Any] = {
        "status": research.get("status", ""),
        "query": research.get("query", ""),
        "meaning": research.get("meaning", ""),
        "requirements": research.get("requirements", []),
        "sensitive_steps": research.get("sensitive_steps", []),
    }
    paths = research.get("official_paths") or research.get("possible_sites") or []
    if isinstance(paths, list):
        out["routes"] = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "domain": item.get("domain", ""),
            }
            for item in paths[:5]
            if isinstance(item, dict)
        ]
    return out


def _task_context_section(task_context: dict[str, Any] | None) -> str:
    if not isinstance(task_context, dict) or not task_context:
        return ""
    triage = task_context.get("triage") if isinstance(task_context.get("triage"), dict) else {}
    research = task_context.get("research_report") if isinstance(task_context.get("research_report"), dict) else {}
    allowed = task_context.get("allowed_actions") or []
    blocked = task_context.get("blocked_actions") or []
    return (
        "[Task triage and safety context]\n"
        f"triage={json.dumps(triage, ensure_ascii=False)}\n"
        f"research_summary={json.dumps(_compact_research_for_goal(research), ensure_ascii=False)}\n"
        f"allowed_actions={json.dumps(allowed, ensure_ascii=False)}\n"
        f"blocked_actions={json.dumps(blocked, ensure_ascii=False)}\n"
        "Proceed automatically only for read, navigation, search, and public information collection. "
        "Before login, personal data, authentication, agreement, application, payment, account, finance, or legal-effect steps, stop and request human confirmation.\n\n"
    )


def _build_site_goal(
    search_keyword: str,
    profile: dict,
    direct_search_url: str = "",
    target_count: int = 0,
    task_context: dict[str, Any] | None = None,
    collection_intent: dict[str, Any] | None = None,
) -> str:
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
    navigation = manual.get("navigation_policy", {}) if isinstance(manual, dict) else {}
    start_url = str(navigation.get("start_url") or base_url or "").strip()
    search_entry = str(navigation.get("search_entry") or "").strip()
    navigation_section = ""
    direct_search_section = ""
    if direct_search_url:
        direct_search_section = (
            "[Code-generated search URL]\n"
            f"{direct_search_url}\n"
            "Use this exact URL with open_browser when direct search navigation is useful. "
            "Do not hand-encode Korean query text.\n\n"
        )
    elif start_url:
        navigation_section = (
            "[Navigation start]\n"
            f"Open only the site home page with open_browser: {start_url}\n"
            "Do not construct or open a search/query URL yourself. "
            "Find the visible search input, type the user query, and submit from the page."
            f"{' ' + search_entry if search_entry else ''}\n\n"
        )
    intent = normalize_collection_intent(
        collection_intent,
        site=str(entry.get("slug") or ""),
        search_keyword=search_keyword,
        target_count=target_count,
    )
    target_section = ""
    if int(target_count or 0) > 0:
        target_section = (
            "[Collection target]\n"
            f"Collect up to {int(target_count)} distinct job postings for this request. "
            "When that many valid detail pages have been collected and submitted, finish instead of opening more cards.\n\n"
        )
    elif intent.count_mode == CollectionCountMode.VISIBLE_ALL:
        target_section = (
            "[Collection target]\n"
            "Collect every relevant job card visible on the first stable search-result screen. "
            "Do not invent a fixed item count and do not continue to additional result pages unless the user explicitly requested it.\n\n"
        )
    filters = dump_model(intent.filters)
    active_filters = {key: value for key, value in filters.items() if value}
    request_section = (
        "[Structured user request]\n"
        f"count_mode={intent.count_mode.value}\n"
        f"filters={json.dumps(active_filters, ensure_ascii=False)}\n"
        f"freshness_required={str(intent.freshness_required).lower()}\n"
        f"purpose={intent.purpose.value}\n"
        f"analysis_goal={intent.analysis_goal}\n"
        "Apply visible site filters when available. Do not claim that a posting meets a date or filter condition without visible evidence. "
        "The worker collects supporting postings; the answer agent performs the final comparison or trend analysis.\n\n"
    )
    task_context_section = _task_context_section(task_context)

    return (
        f"{site_name}({base_url})에서 '{search_keyword}' 채용공고를 검색하고 수집하세요. "
        "검색 결과 화면에 도달하면 공고 수와 카드/행 목록을 확인하고, "
        "방문할 공고별 순회 계획을 세운 뒤 상세 페이지를 하나씩 방문하세요. "
        "상세 페이지에서 필요한 정보를 수집한 뒤 목록으로 돌아와 이미 클릭한 공고는 제외하고 다음 공고를 선택하세요.\n\n"
        f"{navigation_section}"
        f"{direct_search_section}"
        f"{target_section}"
        f"{request_section}"
        f"{task_context_section}"
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



def _commit_feedback_episodes(final_state: dict, hit_recursion_limit: bool, is_finished: bool, run_id: str = "") -> int:
    """Persist feedback episodes for later Critic/Recipe Memory promotion. Best-effort."""
    episodes = list(final_state.get("feedback_episodes", []) or [])
    if not episodes:
        return 0
    run_status = "finished" if is_finished else "recursion_limit" if hit_recursion_limit else "stopped"
    try:
        from agent.recipe.feedback_store import FeedbackStore

        saved = FeedbackStore().commit_episodes(
            episodes,
            run_id=run_id or None,
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


def needs_human_limit_approval(
    *,
    hit_recursion_limit: bool,
    is_finished: bool,
    persisted_count: int,
    target_count: int = 0,
) -> bool:
    if not _env_bool("VISION_HITL_ON_RECURSION_LIMIT", True):
        return False
    persisted = int(persisted_count or 0)
    target = int(target_count or 0)
    if target > 0 and persisted >= target:
        return False
    return bool(hit_recursion_limit and not is_finished and persisted > 0)


def _job_report_items(submission: dict) -> list[dict[str, str]]:
    summary = submission.get("extracted_summary") if isinstance(submission, dict) else {}
    jobs = summary.get("jobs") if isinstance(summary, dict) else []
    out: list[dict[str, str]] = []
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        out.append(
            {
                "company": str(job.get("company") or ""),
                "position": str(job.get("position") or ""),
                "url": str(job.get("url") or ""),
                "field_count": str(job.get("field_count") or ""),
            }
        )
    return out


def build_limit_intermediate_report(
    worker_result: dict,
    submission: dict,
    *,
    persisted_count: int,
    current_limit: int,
    target_count: int = 0,
) -> dict[str, Any]:
    final_state = worker_result.get("final_state") if isinstance(worker_result, dict) else {}
    if not isinstance(final_state, dict):
        final_state = {}
    summary = submission.get("extracted_summary") if isinstance(submission, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    plan = list(final_state.get("plan", []) or [])
    current_step = int(final_state.get("current_plan_step", 0) or 0)
    remaining_plan = plan[current_step: current_step + 4] if plan else []
    suggested_limit = _suggested_recursion_limit(current_limit)
    target = int(target_count or submission.get("target_count") or worker_result.get("target_count") or 0)
    persisted = int(persisted_count or 0)
    return {
        "status": "needs_human_limit_approval",
        "reason": "recursion_limit_reached_with_partial_data",
        "current_recursion_limit": current_limit,
        "suggested_recursion_limit": suggested_limit,
        "collected_count": int(submission.get("collected_count") or summary.get("job_count") or 0),
        "persisted_count": persisted,
        "target_count": target,
        "remaining_collection_count": max(0, target - persisted) if target > 0 else 0,
        "collection_complete": bool(target > 0 and persisted >= target),
        "current_url": str(summary.get("current_url") or final_state.get("current_url") or ""),
        "current_plan_step": current_step,
        "total_plan_steps": len(plan),
        "remaining_plan_preview": [str(item) for item in remaining_plan],
        "jobs": _job_report_items(submission),
        "question": f"현재 recursion limit {current_limit}에 도달했습니다. limit을 {suggested_limit}로 늘려 계속 진행할까요?",
    }


def limit_report_requires_more_collection(report: dict) -> bool:
    """Return True when explicit collection counters say more data is needed."""
    if not isinstance(report, dict):
        return False
    target = int(report.get("target_count") or 0)
    persisted = int(report.get("persisted_count") or 0)
    if target > 0:
        return persisted < target
    return persisted > 0



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
    from agent.graph.state_factory import create_worker_state

    return create_worker_state(goal)




def run_worker_once(
    search_keyword: str,
    site: str | None = None,
    target_count: int = 0,
    task_category: str = DEFAULT_SEARCH_TASK_CATEGORY,
    search_intent_resolved: bool = False,
    review_feedback: str | None = None,
    review_attempt: int = 0,
    run_id: str | None = None,
    task_context: dict[str, Any] | None = None,
    collection_intent: dict[str, Any] | None = None,
) -> dict:
    """Run one child vision worker attempt and return an unreviewed submission payload."""
    from agent.recipe.reviewer import build_worker_submission, new_worker_run_id

    site_profile = _load_collection_profile(site)
    site_entry = site_profile["entry"]
    site_slug = site_entry.get("slug") or site or "unknown"
    site_name = site_entry.get("display_name") or site_slug
    run_id = run_id or new_worker_run_id()
    raw_search_keyword = search_keyword
    requested_target_count = _normalize_target_count(target_count)
    if collection_intent:
        search_intent = dump_model(
            normalize_collection_intent(
                collection_intent,
                original_query=raw_search_keyword,
                site=site_slug,
                search_keyword=search_keyword,
                target_count=requested_target_count,
            )
        )
        search_intent["source"] = "structured_arguments"
        search_intent["error"] = ""
    elif search_intent_resolved:
        search_intent = dump_model(
            normalize_collection_intent(
                original_query=raw_search_keyword,
                site=site_slug,
                search_keyword=search_keyword,
                target_count=requested_target_count,
            )
        )
        search_intent["source"] = "structured_arguments"
        search_intent["error"] = ""
    else:
        search_intent = _extract_search_intent(search_keyword, site_profile)
    search_keyword = str(search_intent.get("search_keyword") or search_keyword or "").strip()
    inferred_target_count = _normalize_target_count(search_intent.get("target_count") or 0)
    target_count = requested_target_count or inferred_target_count
    task_category = normalize_task_category(task_category or DEFAULT_SEARCH_TASK_CATEGORY)
    direct_search_url = _build_direct_search_url(search_keyword, site_profile)
    goal = _append_review_feedback(
        _build_site_goal(
            search_keyword,
            site_profile,
            direct_search_url,
            target_count=target_count,
            task_context=task_context,
            collection_intent=search_intent,
        ),
        review_feedback,
    )
    initial_state = _initial_worker_state(goal)
    initial_state["recipe_params"] = {
        "query": search_keyword,
        "keyword": search_keyword,
        "target_count": target_count,
        "site": site_slug,
        "task_category": task_category,
        "count_mode": search_intent.get("count_mode", CollectionCountMode.UNSPECIFIED.value),
        "collection_intent": search_intent,
    }

    logger.info(
        "[realtime_scraping] Starting worker graph site=%s attempt=%s",
        site_slug,
        review_attempt,
    )
    recursion_limit = int(os.getenv("VISION_AGENT_RECURSION_LIMIT", str(DEFAULT_RECURSION_LIMIT)))
    final_state, hit_recursion_limit = _execute_worker_graph(
        initial_state,
        site_profile,
        recursion_limit,
        prepare_screen=_prepare_worker_start_screen,
        run_graph=_run_graph_with_last_state,
    )
    from agent.application.run_context import record_graph_state_metrics

    record_graph_state_metrics(final_state)

    extracted = final_state.get("extracted_jd", {}) or {}
    is_finished = bool(final_state.get("is_finished", False))
    run_status = _worker_run_status(hit_recursion_limit, is_finished)
    feedback_saved = _commit_feedback_episodes(final_state, hit_recursion_limit, is_finished, run_id=run_id)
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
        target_count=target_count,
        task_category=task_category,
    )

    return {
        "submission": submission,
        "extracted_jd": extracted,
        "final_state": final_state,
        "site_slug": site_slug,
        "site_name": site_name,
        "keyword": search_keyword,
        "raw_keyword": raw_search_keyword,
        "target_count": target_count,
        "task_category": task_category,
        "search_intent": search_intent,
        "collection_intent": search_intent,
        "task_context": task_context or {},
        "run_status": run_status,
        "hit_recursion_limit": hit_recursion_limit,
        "is_finished": is_finished,
        "feedback_saved": feedback_saved,
        "recursion_limit": recursion_limit,
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


def _recipe_learning_mode() -> str:
    mode = os.getenv("VISION_RECIPE_LEARNING_MODE", "record").strip().lower()
    # worker 실행 중에는 후보 저장까지만 한다. review/promote는 별도 후처리 도구에서 수행한다.
    return mode if mode in {"off", "record"} else "record"


def _commit_recipe_candidate(
    submission: dict,
    review: dict,
    source: str,
    submission_id: str,
    mode: str,
) -> str:
    """학습 모드에 따라 Reflex 레시피 후보를 저장한다."""
    if mode == "off" or not review.get("recipe_candidate"):
        return ""
    try:
        from agent.recipe.candidate_store import RecipeCandidateStore

        candidate_id = RecipeCandidateStore().commit_candidate(
            submission,
            review=review,
            source=source,
            submission_id=submission_id,
        )
        if candidate_id:
            logger.info("[realtime_scraping] Recipe candidate stored: id=%s mode=%s", candidate_id, mode)
        return candidate_id
    except Exception as e:
        logger.debug("[realtime_scraping] Recipe candidate commit skipped: %s", e)
        return ""


def _schedule_recipe_candidate_promotion(candidate_id: str) -> bool:
    """작업자 실행과 분리된 후처리 서비스에 후보 승격을 맡긴다."""

    from agent.application.recipe_promotion_service import schedule_recipe_candidate_promotion

    return schedule_recipe_candidate_promotion(candidate_id)


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
    learning_mode = _recipe_learning_mode()
    if learning_mode != "off":
        candidate_id = _commit_recipe_candidate(submission, review, source, submission_id, learning_mode)
        if candidate_id:
            submission["recipe_candidate_id"] = candidate_id
            submission["recipe_learning_mode"] = learning_mode
            _schedule_recipe_candidate_promotion(candidate_id)
    return persisted_count, submission, review, submission_id


def _result_payload(
    *,
    message: str,
    site_name: str,
    site_slug: str,
    keyword: str,
    item_count: int,
    persisted_count: int,
    target_count: int = 0,
    submission_id: str,
    review: dict,
    hit_recursion_limit: bool,
    is_finished: bool,
    needs_human_approval: bool = False,
    intermediate_report: dict | None = None,
    task_category: str = "",
) -> str:
    return json.dumps(
        {
            "message": message,
            "site": site_slug,
            "site_name": site_name,
            "keyword": keyword,
            "target_count": int(target_count or 0),
            "task_category": normalize_task_category(task_category),
            "item_count": item_count,
            "persisted_count": persisted_count,
            "submission_id": submission_id,
            "review": review,
            "hit_recursion_limit": hit_recursion_limit,
            "is_finished": is_finished,
            "needs_human_approval": needs_human_approval,
            "intermediate_report": intermediate_report or {},
        },
        ensure_ascii=False,
        indent=2,
    )


def _run_realtime_scraping(
    company: str = None,
    tech_stack: str = None,
    site: str = None,
    query: str = None,
    target_count: int = 0,
    task_category: str = DEFAULT_SEARCH_TASK_CATEGORY,
    search_intent_resolved: bool = False,
    review_feedback: str = None,
    review_attempt: int = 0,
    count_mode: str = CollectionCountMode.UNSPECIFIED.value,
    posted_date_expression: str = "",
    posted_from: str = "",
    posted_to: str = "",
    experience: str = "",
    location: str = "",
    employment_type: str = "",
    freshness_required: bool = False,
    purpose: str = "collect",
    analysis_goal: str = "",
    original_query: str = "",
) -> str:
    """
    Run the child vision worker, review its structured submission, and persist only accepted data.
    task_category는 Reflex 증거에 저장되어 반복 실행 시 같은 작업 유형의 레시피를 고르는 데 사용된다.
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

    from agent.application.collection_service import (
        CollectionOperations,
        CollectionRequest,
        CollectionService,
    )
    from agent.recipe.reviewer import render_review_feedback

    logger.info("[realtime_scraping] Invoking vision worker for keyword=%r", search_keyword)
    operations = CollectionOperations(
        normalize_target_count=_normalize_target_count,
        normalize_task_category=normalize_task_category,
        review_retries=_worker_review_retries,
        run_worker=run_worker_once,
        review_worker=commit_worker_review,
        persist_result=persist_accepted_worker_result,
        render_review_feedback=render_review_feedback,
        needs_approval=needs_human_limit_approval,
        build_intermediate_report=build_limit_intermediate_report,
        report_requires_more_collection=limit_report_requires_more_collection,
        close_browser=_close_browser_after_run,
    )
    collection_intent = normalize_collection_intent(
        {
            "original_query": original_query or query or search_keyword,
            "site": site or "",
            "search_keyword": search_keyword,
            "count_mode": count_mode,
            "target_count": target_count,
            "filters": {
                "posted_date_expression": posted_date_expression,
                "posted_from": posted_from,
                "posted_to": posted_to,
                "experience": experience,
                "location": location,
                "employment_type": employment_type,
            },
            "freshness_required": freshness_required,
            "purpose": purpose,
            "analysis_goal": analysis_goal,
        }
    )
    result = CollectionService(operations).collect(
        CollectionRequest(
            search_keyword=search_keyword,
            site=site,
            target_count=target_count,
            task_category=task_category or DEFAULT_SEARCH_TASK_CATEGORY,
            search_intent_resolved=search_intent_resolved,
            review_feedback=review_feedback or "",
            review_attempt=review_attempt,
            collection_intent=dump_model(collection_intent),
        )
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def realtime_scraping(
    company: str = None,
    tech_stack: str = None,
    site: str = None,
    query: str = None,
    target_count: int = 0,
    task_category: str = DEFAULT_SEARCH_TASK_CATEGORY,
    review_feedback: str = None,
    review_attempt: int = 0,
    count_mode: str = CollectionCountMode.UNSPECIFIED.value,
    posted_date_expression: str = "",
    posted_from: str = "",
    posted_to: str = "",
    experience: str = "",
    location: str = "",
    employment_type: str = "",
    freshness_required: bool = False,
    purpose: str = "collect",
    analysis_goal: str = "",
    original_query: str = "",
) -> str:
    """구조화된 요청에 따라 비전 작업자로 공고를 수집하고 승인된 결과만 저장한다."""

    from agent.application.run_context import emit_run_event, run_context
    from agent.application.run_contracts import RunPhase, RunStatus

    context_query = query or " ".join(
        part for part in (company, tech_stack) if part
    )
    with run_context(query=context_query, prefix="collection") as (context, created):
        with worker_execution_session():
            result_text = _run_realtime_scraping(
                company=company,
                tech_stack=tech_stack,
                site=site,
                query=query,
                target_count=target_count,
                task_category=task_category,
                search_intent_resolved=_normalize_target_count(target_count) > 0,
                review_feedback=review_feedback,
                review_attempt=review_attempt,
                count_mode=count_mode,
                posted_date_expression=posted_date_expression,
                posted_from=posted_from,
                posted_to=posted_to,
                experience=experience,
                location=location,
                employment_type=employment_type,
                freshness_required=freshness_required,
                purpose=purpose,
                analysis_goal=analysis_goal,
                original_query=original_query,
            )
        if not created:
            return result_text

        try:
            payload = json.loads(result_text)
        except (TypeError, json.JSONDecodeError):
            payload = {"message": str(result_text)}
        payload["run_id"] = context.run_id
        payload["metrics"] = context.snapshot()
        failed = str(payload.get("message") or "").startswith("collection error")
        emit_run_event(
            "run_failed" if failed else "run_completed",
            RunPhase.FAILED if failed else RunPhase.COMPLETED,
            "수집 작업이 실패했습니다." if failed else "수집 작업을 완료했습니다.",
            status=RunStatus.FAILED if failed else RunStatus.COMPLETED,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)
