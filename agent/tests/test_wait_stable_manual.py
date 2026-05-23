import time
from agent.tools.perception import PerceptionEngine
from agent.utils.wait_stable import WaitStable
from agent.utils.logger import logger

def main():
    logger.info("Starting WaitStable test...")
    
    engine = PerceptionEngine()
    waiter = WaitStable(engine)
    
    logger.info("Testing stable screen (No movement expected)...")
    # 화면을 가만히 두었을 때 곧바로 안정화(True) 반환하는지 테스트
    result = waiter.wait(max_wait_sec=5.0)
    logger.info("Stable test result", result=result)

if __name__ == "__main__":
    main()
