import time
import pyautogui
import sys
import os

# 프로젝트 루트 경로를 sys.path에 추가하여 'agent' 모듈을 찾을 수 있게 합니다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from agent.tools.perception import PerceptionEngine
from agent.utils.logger import logger

def test_coordinate_accuracy():
    """
    PerceptionEngine으로 캡처한 브라우저 화면의 중앙 좌표를 계산하고,
    마우스가 해당 좌표로 정확히 이동하여 클릭하는지 시각적으로 검증합니다.
    """
    logger.info("Starting coordinate verification test...")
    
    # 1. 화면 캡처 및 브라우저 영역 파악
    perception = PerceptionEngine()
    region = perception._get_browser_region()
    
    if not region:
        logger.error("Browser window not found. Please open Chrome/Edge/Whale.")
        return
        
    logger.info(f"Browser region found: {region}")
    
    # 2. 브라우저 정중앙 절대 좌표 계산
    center_x = region["left"] + (region["width"] // 2)
    center_y = region["top"] + (region["height"] // 2)
    
    logger.info(f"Calculated center absolute coordinates: ({center_x}, {center_y})")
    
    # 3. 마우스 이동 (0.2초)
    logger.info("Moving mouse rapidly...")
    
    # 4. 마우스 이동 및 클릭
    pyautogui.moveTo(center_x, center_y, duration=0.2)
    pyautogui.click()
    logger.info("Test completed successfully.")

if __name__ == "__main__":
    test_coordinate_accuracy()
