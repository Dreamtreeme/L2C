from agent.tools.perception import PerceptionEngine
from agent.utils.logger import logger
import os

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
