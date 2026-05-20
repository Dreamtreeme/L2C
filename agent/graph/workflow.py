from langgraph.graph import StateGraph, START, END

from agent.graph.state import GraphState
from agent.graph.nodes import perception_node, reasoning_node, action_node
from agent.utils.logger import logger

def should_continue(state: GraphState) -> str:
    """다음으로 이동할 노드를 결정하는 라우팅 함수입니다."""
    if state.get("is_finished", False):
        logger.info("Task marked as finished. Ending workflow.")
        return "end"
        
    if state.get("error_count", 0) >= 3:
        logger.error("Too many errors. Forcing workflow to end.")
        return "end"
        
    return "perception"

def build_graph():
    """LangGraph 워크플로우를 구성하고 컴파일된 앱을 반환합니다."""
    
    logger.info("Building StateGraph workflow...")
    
    # 1. StateGraph 초기화
    workflow = StateGraph(GraphState)
    
    # 2. 노드 추가
    workflow.add_node("perception", perception_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("action", action_node)
    
    # 3. 엣지 연결 (흐름 정의)
    # 시작 시 빈 계획 상태로 reasoning 노드로 진입하여 동적 계획 수립 유도
    workflow.add_edge(START, "reasoning")
    
    # perception 완료 후 reasoning으로 이동
    workflow.add_edge("perception", "reasoning")
    
    # reasoning 완료 후 action으로 이동
    workflow.add_edge("reasoning", "action")
    
    # action 완료 후 조건부 라우팅 (계속 진행할지 종료할지)
    workflow.add_conditional_edges(
        "action",
        should_continue,
        {
            "perception": "perception",
            "end": END
        }
    )
    
    # 4. 그래프 컴파일
    app = workflow.compile()
    logger.info("StateGraph compiled successfully.")
    
    return app
