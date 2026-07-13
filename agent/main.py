import sys
import os

# Add project root to sys.path to allow execution from outside or inside the folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.application.chat_service import get_chat_service
from agent.application.run_contracts import new_run_id

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
    
    try:
        run_id = new_run_id("cli")
        result = get_chat_service().run(query, run_id=run_id)
        final_answer = str(result.get("last_action_result") or "")
        
        print("\n==========================================")
        print("💡 지휘자 최종 답변:")
        print("==========================================")
        print(final_answer)
        print(f"실행 식별자: {run_id}")
        print("==========================================\n")
        
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
