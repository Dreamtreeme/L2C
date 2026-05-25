import os
import sys
# 프로젝트 루트 경로를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agent.tools.perception import PerceptionEngine
from agent.utils.logger import logger

def main():
    logger.info("Starting Perception Engine test...")
    
    # 1. PerceptionEngine 초기화
    engine = PerceptionEngine()
    
    # 2. 화면 캡처 테스트
    logger.info("Capturing screen...")
    screenshot_path = engine.capture_screen("test_capture.png")
    
    if screenshot_path.exists():
        logger.info("Screen capture successful!", path=str(screenshot_path))
    else:
        logger.error("Screen capture failed to save file.")
        return

    # 3. UI 파싱(Mock) 테스트
    logger.info("Testing UI Analysis (Mock)...")
    result = engine.analyze_ui(screenshot_path)
    
    logger.info("Analysis Result", markers=result.get("markers"))

if __name__ == "__main__":
    main()
