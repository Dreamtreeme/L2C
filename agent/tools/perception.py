import datetime
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mss
import mss.tools
import pygetwindow as gw
from PIL import Image

from agent.utils.logger import logger
from shared.config import SCREENSHOT_DIR


class PerceptionEngine:
    """
    모니터 화면을 인식하고 분석하는 Perception 엔진입니다.
    mss를 이용한 고속 화면 캡처 및 (추후) OmniParser 연동을 담당합니다.
    """

    def __init__(self):
        self.screenshot_dir = SCREENSHOT_DIR
        self.sct = mss.mss()
        logger.info("PerceptionEngine initialized", screenshot_dir=str(self.screenshot_dir))

    def _get_browser_region(self) -> Optional[Dict[str, int]]:
        """
        열려있는 창 중에서 브라우저(Chrome, Edge, Whale)를 찾아 해당 영역을 반환합니다.
        """
        keywords = ["Chrome", "Edge", "Whale", "크롬", "엣지", "웨일"]
        
        for win in gw.getAllWindows():
            # 최소화되어 있거나 숨겨진 창은 제외
            if not win.visible or win.isMinimized:
                continue
                
            if any(k in win.title for k in keywords):
                # Windows 10/11의 DWM(Desktop Window Manager)은 
                # 창 주변의 투명한 그림자 영역(약 8px)까지 창 크기로 인식합니다.
                # 배경이 찍히는 것을 막기 위해 이 보이지 않는 테두리를 잘라냅니다.
                border = 8
                
                # 최대화 상태일 때 상단 여백도 조정 필요 (-8로 넘어오는 경우가 많음)
                top = win.top + border if win.isMaximized else win.top
                
                return {
                    "top": top,
                    "left": win.left + border,
                    "width": win.width - (border * 2),
                    "height": win.height - border - (border if win.isMaximized else 0)
                }
        return None

    def capture_screen(self, filename: Optional[str] = None) -> Path:
        """
        브라우저 창 영역(없으면 주 모니터 전체)을 캡처하여 지정된 디렉토리에 저장합니다.
        
        Args:
            filename: 저장할 파일명. 입력하지 않으면 타임스탬프 기반 자동 생성.
            
        Returns:
            저장된 스크린샷 이미지의 절대 경로 (Path 객체)
        """
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screen_{timestamp}.png"
            
        output_path = self.screenshot_dir / filename
        
        # 1. 브라우저 창 영역 찾기
        region = self._get_browser_region()
        
        try:
            if region:
                # 브라우저만 캡처
                sct_img = self.sct.grab(region)
                logger.debug("Captured browser window only", region=region)
            else:
                # 브라우저를 못 찾으면 모니터 1번 (주 모니터) 전체 캡처
                monitor = self.sct.monitors[1]
                sct_img = self.sct.grab(monitor)
                logger.debug("Browser not found, captured full monitor", monitor=monitor)
                
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=str(output_path))
            
            logger.info(
                "Screen captured successfully", 
                width=sct_img.width, 
                height=sct_img.height, 
                output_path=str(output_path)
            )
            return output_path
        except Exception as e:
            logger.exception("Failed to capture screen", error=str(e))
            raise

    def analyze_ui(self, image_path: Path) -> Dict[str, Any]:
        """
        캡처된 이미지를 기반으로 UI 요소를 파싱합니다.
        현재는 Phase 2 초안으로 Mock 응답을 반환하며, 향후 OmniParser API 연동으로 교체됩니다.
        
        Args:
            image_path: 파싱할 이미지 파일의 경로
            
        Returns:
            UI 마커의 ID, 텍스트, 바운딩 박스(bbox) 목록을 담은 딕셔너리
        """
        logger.info("Analyzing UI elements", image_path=str(image_path))
        
        if not image_path.exists():
            logger.error("Image file not found for UI analysis", image_path=str(image_path))
            raise FileNotFoundError(f"Image not found: {image_path}")

        # TODO: 로컬 또는 원격 OmniParser 서버에 HTTP POST 요청하여 요소 추출
        # 여기서는 동작 테스트를 위해 가짜(Mock) 데이터를 반환합니다.
        mock_result = {
            "markers": [
                {"id": 0, "text": "검색창", "bbox": [100, 50, 400, 80]}, # x_min, y_min, x_max, y_max
                {"id": 1, "text": "검색 버튼", "bbox": [410, 50, 460, 80]},
                {"id": 2, "text": "상세 정보 더 보기", "bbox": [500, 800, 700, 850]}
            ],
            "original_image": str(image_path)
        }
        
        logger.debug("UI analysis completed (Mock)", markers_count=len(mock_result["markers"]))
        return mock_result
