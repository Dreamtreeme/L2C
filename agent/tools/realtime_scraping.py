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


def _completed_partial_recorded_steps(final_state: dict) -> list[dict]:
    """Keep only completed UI collection segments; drop the unfinished tail that hit recursion_limit."""
    recorded_steps = list(final_state.get("recorded_steps", []) or [])
    last_return_index = -1
    for idx, step in enumerate(recorded_steps):
        if isinstance(step, dict) and step.get("action") == "go_back":
            last_return_index = idx
    if last_return_index < 0:
        return []
    return recorded_steps[: last_return_index + 1]


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
        logger.info(f"[realtime_scraping] Browser cleanup completed: {result}")
    except Exception as e:
        logger.debug(f"[realtime_scraping] Browser cleanup skipped: {e}")


def _commit_reflex_recipe_for_partial_success(
    final_state: dict,
    persisted_count: int,
    hit_recursion_limit: bool,
    is_finished: bool,
) -> None:
    if not hit_recursion_limit or is_finished or persisted_count <= 0:
        return
    recorded_steps = _completed_partial_recorded_steps(final_state)
    if not recorded_steps:
        logger.info("[realtime_scraping] Reflex partial recipe commit skipped: no completed segment")
        return
    try:
        from agent.recipe import record

        record.commit_if_finished(recorded_steps, final_state, final_state.get("current_url", ""))
        logger.info(
            "[realtime_scraping] Reflex recipe committed from partial success: steps=%s, persisted_count=%s",
            len(recorded_steps),
            persisted_count,
        )
    except Exception as e:
        logger.debug(f"[realtime_scraping] Reflex partial recipe commit skipped: {e}")

@tool
def realtime_scraping(company: str = None, tech_stack: str = None, site: str = None, query: str = None) -> str:
    """
    비전 기반 자율 에이전트를 구동하여 특정 기업(company), 기술 스택(tech_stack), 자유 검색어(query)에 맞는
    최신 채용 공고를 실시간으로 수집하고 데이터베이스에 적재하는 도구입니다.
    내부적으로 SoM(Set-of-Mark) 마커 기반 화면 인식 → LLM 추론 → 물리 조작의
    자율 수집 LangGraph 워크플로우를 기동하며, 모든 도구 호출 궤적이
    LangSmith에 저장되어 Playwright 스크립트 자동 생성의 학습 데이터로 환원됩니다.
    데이터베이스에 정보가 없거나 부족할 때 호출되어 실시간 비전 수집으로 DB를 동적으로 보강합니다.
    """
    # 검색 키워드 조합
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
        return "수집 실패: 검색 키워드(company, tech_stack, query)가 모두 누락되었습니다."

    logger.info(f"[realtime_scraping] Invoking vision agent graph for keyword: '{search_keyword}'")

    try:
        # 비전 자율 수집 LangGraph 워크플로우 빌드 및 기동
        from agent.graph.workflow import build_graph

        app = build_graph()

        # 검색 키워드와 사이트 프로필을 기반으로 자율 수집 목표(goal) 구성
        site_profile = _load_collection_profile(site)
        site_entry = site_profile["entry"]
        site_name = site_entry.get("display_name") or site_entry.get("slug")
        goal = _build_site_goal(search_keyword, site_profile)

        # 초기 상태 구성 (GraphState 스키마에 맞춤)
        initial_state = {
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
            "reflex_state_key": "",
            "reflex_hit": False,
            "reflex_expected_next_state": "",
            "reflex_pending_validation": False,
            "ocr_texts": [],
            "ocr_delta_added": [],
            "ocr_delta_removed": [],
            "reflex_validation_status": "",
        }

        logger.info(f"[realtime_scraping] Starting autonomous vision collection graph for site={site_entry.get('slug')} with goal: {goal}")

        recursion_limit = int(os.getenv("VISION_AGENT_RECURSION_LIMIT", str(DEFAULT_RECURSION_LIMIT)))

        # LangGraph 앱 실행 (동기 stream)
        # 모든 perception → reasoning → action 루프가 자율적으로 순환하며
        # 도구 호출 궤적이 LangSmith 트레이스에 자동 기록됨
        final_state, hit_recursion_limit = _run_graph_with_last_state(app, initial_state, recursion_limit)

        # 수집 결과 분석
        collected = final_state.get("collected_data", [])
        extracted = final_state.get("extracted_jd", {})
        is_finished = final_state.get("is_finished", False)

        if extracted:
            # 수집된 데이터를 DB에 전처리 및 적재
            persisted_count = _persist_collected_data(extracted, search_keyword)
            _commit_reflex_recipe_for_partial_success(
                final_state,
                persisted_count,
                hit_recursion_limit,
                is_finished,
            )

            item_count = len(extracted.get("공고목록", [])) if isinstance(extracted.get("공고목록"), list) else 1
            logger.info(
                f"[realtime_scraping] Vision agent collection completed. "
                f"Items: {item_count}, persisted: {persisted_count}, finished: {is_finished}"
            )
            completion_type = "부분 수집 및 적재 완료" if hit_recursion_limit and not is_finished else "실시간 비전 자율 수집 및 적재 완료"
            return (
                f"{completion_type}: '{search_keyword}' 키워드로 "
                f"{site_name}에서 총 {item_count}건 중 {persisted_count}건의 채용 공고 정보가 데이터베이스에 업데이트되었습니다."
            )
        if hit_recursion_limit:
            return f"실시간 수집 중단: {site_name}에서 '{search_keyword}' 수집 중 recursion limit에 도달했고 저장 가능한 공고가 없습니다."
        else:
            logger.warning(f"[realtime_scraping] Vision agent finished but no data collected for '{search_keyword}'")
            return f"실시간 수집 완료: {site_name}에서 '{search_keyword}'에 매칭되는 유효한 채용 정보를 찾지 못했습니다."

    except Exception as e:
        logger.error(f"[realtime_scraping] Vision agent execution error: {e}", exc_info=True)
        return f"실시간 수집 오류: {e}"
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
