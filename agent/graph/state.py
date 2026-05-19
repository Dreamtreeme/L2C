from typing import Dict, Any, List, Optional
from pathlib import Path

from agent.utils.logger import logger

class AgentState:
    """
    에이전트의 워크플로우 상태를 관리하는 클래스입니다.
    최근 캡처한 이미지, 인식된 마커들, 행동 이력 등을 보관합니다.
    """
    def __init__(self, max_history: int = 10, max_images: int = 2):
        self.max_history = max_history
        self.max_images = max_images
        
        # 상태 저장 공간
        self.action_history: List[Dict[str, Any]] = []
        self.recent_images: List[Path] = []
        
        # 현재 화면 상태
        self.current_markers: List[Dict[str, Any]] = []
        self.last_parsed_text: Optional[str] = None
        
        # 수집 진행 상태
        self.collected_data: List[Any] = []
        
    def add_image(self, image_path: Path) -> None:
        """최근 캡처한 이미지 경로를 추가하고 오래된 것은 관리합니다."""
        self.recent_images.append(image_path)
        if len(self.recent_images) > self.max_images:
            # 오래된 이미지 경로는 리스트에서만 제거 (실제 파일 삭제는 별도 정책에 따름)
            old_image = self.recent_images.pop(0)
            logger.debug(f"Removed old image from state: {old_image}")
            
    def set_current_markers(self, markers: List[Dict[str, Any]]) -> None:
        """현재 화면에서 파싱된 마커 목록을 업데이트합니다."""
        self.current_markers = markers
        
    def add_action_history(self, action_result: Dict[str, Any]) -> None:
        """실행한 행동의 결과를 이력에 추가합니다."""
        self.action_history.append(action_result)
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
            
    def get_last_action(self) -> Optional[Dict[str, Any]]:
        """가장 최근에 실행한 행동 결과를 반환합니다."""
        if not self.action_history:
            return None
        return self.action_history[-1]
        
    def get_summary(self) -> Dict[str, Any]:
        """현재 상태의 요약을 반환합니다 (LLM 프롬프트에 주입하기 유용)."""
        return {
            "recent_actions": [a.get("action") for a in self.action_history[-3:]],
            "markers_count": len(self.current_markers),
            "collected_items": len(self.collected_data)
        }
