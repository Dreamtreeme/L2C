import os

from langgraph.graph import StateGraph, START, END

from agent.graph.state import GraphState
from agent.graph.nodes import perception_node, reflex_node, reasoning_node, action_node
from agent.utils.logger import logger

def should_continue(state: GraphState) -> str:
    """다음으로 이동할 노드를 결정하는 라우팅 함수입니다."""
    if state.get("is_finished", False):
        logger.info("Task marked as finished. Ending workflow.")
        return "end"
    if state.get("pending_human_approval", False):
        logger.info("Human approval required. Ending worker loop.")
        return "end"
        
    if state.get("error_count", 0) >= 3:
        logger.error("Too many errors. Forcing workflow to end.")
        return "end"

    if state.get("last_action_screen_changed") is False:
        logger.info("Last action did not change the screen. Skipping perception.")
        return "reasoning"

    return "perception"


def _reflex_enabled() -> bool:
    return os.getenv("REFLEX_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}


def route_after_perception(state: GraphState) -> str:
    """전환 계약 판정 뒤 재관찰, reflex 재생 또는 reasoning 폴백을 선택한다."""
    if state.get("low_information_screen"):
        retry_count = int(state.get("low_information_retry_count", 0) or 0)
        try:
            retry_limit = max(1, int(os.getenv("VISION_LOW_INFORMATION_MAX_CYCLES", "3")))
        except ValueError:
            retry_limit = 3
        if retry_count < retry_limit:
            logger.info(
                "Low-information screen will be observed again without reasoning.",
                retry_count=retry_count,
                retry_limit=retry_limit,
            )
            return "perception"
        logger.info(
            "Low-information retry limit reached; allowing reasoning recovery.",
            retry_count=retry_count,
            retry_limit=retry_limit,
        )
        return "reasoning"

    if state.get("queue_replay_hit"):
        logger.info("Result card queue hit; executing cached next-card action.")
        return "action"
    if state.get("page_policy_hit"):
        logger.info("Page policy hit; executing deterministic detail action.")
        return "action"

    transition_status = state.get("transition_status", "")
    if transition_status == "pending":
        logger.info("Transition still pending; observing the screen again.")
        return "perception"
    if transition_status == "unknown" and state.get("transition_source") in {"reflex", "page_policy"}:
        logger.info(
            "Transition contract could not verify the screen; falling back to reasoning.",
            source=state.get("transition_source", ""),
        )
        return "reasoning"

    if not _reflex_enabled():
        return "reasoning"

    return "reflex"


def route_after_reflex(state: GraphState) -> str:
    """reflex hit이면 action_node로, miss면 reasoning_node로 보냅니다."""
    return "action" if state.get("reflex_hit") else "reasoning"


def route_after_start(state: GraphState) -> str:
    """이미 관찰된 시작 화면이 있으면 첫 판단도 Reflex를 먼저 시도한다."""
    if state.get("low_information_screen"):
        return "perception"
    if not _reflex_enabled():
        return "reasoning"
    if state.get("current_markers") and state.get("current_page_role") and state.get("recent_images"):
        logger.info("Start screen is already observed; trying Reflex before reasoning.")
        return "reflex"
    return "reasoning"


def build_graph():
    """LangGraph 워크플로우를 구성하고 컴파일된 앱을 반환합니다."""
    
    logger.info("Building StateGraph workflow...")
    
    # 1. StateGraph 초기화
    workflow = StateGraph(GraphState)
    
    # 2. 노드 추가
    workflow.add_node("perception", perception_node)
    workflow.add_node("reflex", reflex_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("action", action_node)
    
    # 3. 엣지 연결 (흐름 정의)
    # 시작 화면을 이미 perception으로 준비한 worker는 첫 행동도 Reflex를 먼저 시도한다.
    workflow.add_conditional_edges(
        START,
        route_after_start,
        {
            "perception": "perception",
            "reflex": "reflex",
            "reasoning": "reasoning",
        },
    )
    
    # perception 완료 후 REFLEX_ENABLED일 때만 캐시 재생을 먼저 시도
    workflow.add_conditional_edges(
        "perception",
        route_after_perception,
        {
            "action": "action",
            "perception": "perception",
            "reflex": "reflex",
            "reasoning": "reasoning",
        }
    )

    # reflex miss는 reasoning 폴백, hit는 기존 action_node 실행 경로 재사용
    workflow.add_conditional_edges(
        "reflex",
        route_after_reflex,
        {
            "action": "action",
            "reasoning": "reasoning",
        }
    )
    
    # reasoning 완료 후 action으로 이동
    workflow.add_edge("reasoning", "action")
    
    # action 완료 후 조건부 라우팅 (계속 진행할지 종료할지)
    workflow.add_conditional_edges(
        "action",
        should_continue,
        {
            "perception": "perception",
            "reasoning": "reasoning",
            "end": END
        }
    )
    
    # 4. 그래프 컴파일
    app = workflow.compile()
    logger.info("StateGraph compiled successfully.")
    
    return app
