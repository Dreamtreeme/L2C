import sys
import os

# Add project root to sys.path to allow execution from outside or inside the folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph.nodes import qa_reasoning_node
from agent.graph.state import GraphState

def main():
    if len(sys.argv) < 2:
        print("사용법: python agent/main.py \"[질문 또는 명령]\"")
        sys.exit(1)
        
    query = sys.argv[1].strip()
    if not query:
        print("질문이 비어 있습니다.")
        sys.exit(1)
        
    # Reconfigure stdout/stderr to utf-8 for Windows compatibility and avoiding encoding issues
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    print("\n==========================================")
    print("🤖 L2C 지휘자 에이전트 기동")
    print(f"Goal: {query}")
    print("==========================================\n")
    
    # Initialize state with required typing matching GraphState
    state = GraphState(
        goal=query,
        ui_context="",
        current_markers=[],
        action_history=[],
        recent_images=[],
        marked_image="",
        error_count=0,
        is_finished=False,
        collected_data=[],
        extracted_jd={},
        last_action_result=None,
        plan=[],
        current_plan_step=0
    )
    
    try:
        # Invoke the main commander node
        result = qa_reasoning_node(state)
        final_answer = result.get("last_action_result", "")
        
        print("\n==========================================")
        print("💡 지휘자 최종 답변:")
        print("==========================================")
        print(final_answer)
        print("==========================================\n")
        
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
