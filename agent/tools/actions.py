import time
from typing import Dict, Any, List
import platform

import pyautogui
import pyperclip

from agent.utils.logger import logger
from agent.utils.wait_stable import WaitStable
from agent.tools.perception import PerceptionEngine

class ActionTools:
    """
    물리적인 마우스/키보드 조작을 담당하는 Action 도구 모음입니다.
    """
    
    def __init__(self, perception_engine: PerceptionEngine):
        self.perception = perception_engine
        self.wait_stable = WaitStable(perception_engine)
        
        # pyautogui 기본 안전 설정
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1  # 기본 0.5초에서 0.1초로 단축하여 속도 향상
        
    def _execute_with_wait(self, action_name: str, func, *args, **kwargs) -> Dict[str, Any]:
        """
        액션을 실행하기 전후로 화면 안정화 및 로깅을 처리하는 Wrapper
        """
        logger.info(f"Executing action: {action_name}")
        
        try:
            # 행동 실행
            result = func(*args, **kwargs)
            
            # 행동 직후 화면이 안정될 때까지 대기
            is_stable = self.wait_stable.wait()
            
            logger.info(f"Action '{action_name}' completed", stable=is_stable)
            return {
                "status": "success",
                "action": action_name,
                "result": result,
                "stable": is_stable
            }
        except Exception as e:
            logger.exception(f"Action '{action_name}' failed", error=str(e))
            return {
                "status": "error",
                "action": action_name,
                "error": str(e)
            }
            
    def _get_absolute_coords(self, bbox: List[int]) -> tuple[int, int]:
        """
        상대적인 bbox 좌표를 현재 브라우저 영역의 절대 좌표로 변환하고
        해당 박스의 정중앙을 반환합니다.
        bbox: [x_min, y_min, x_max, y_max] (브라우저 창 내부 상대 좌표라고 가정)
        """
        region = self.perception._get_browser_region()
        if not region:
            raise ValueError("Browser window not found")
            
        x_center_relative = (bbox[0] + bbox[2]) // 2
        y_center_relative = (bbox[1] + bbox[3]) // 2
        
        x_absolute = region["left"] + x_center_relative
        y_absolute = region["top"] + y_center_relative
        
        return x_absolute, y_absolute

    def click_marker(self, bbox: List[int]) -> Dict[str, Any]:
        """마커(UI 요소)의 중앙을 클릭합니다."""
        def _click():
            x, y = self._get_absolute_coords(bbox)
            pyautogui.moveTo(x, y, duration=0.2)
            pyautogui.click()
            return f"Clicked at ({x}, {y})"
            
        return self._execute_with_wait("click_marker", _click)
        
    def type_in_marker(self, bbox: List[int], text: str) -> Dict[str, Any]:
        """마커를 클릭한 후, pyperclip을 통해 안전하게 한글/영문 텍스트를 붙여넣습니다."""
        def _type():
            x, y = self._get_absolute_coords(bbox)
            pyautogui.moveTo(x, y, duration=0.2)
            pyautogui.click()
            
            # 클립보드를 통한 한글 씹힘 방지 타이핑
            pyperclip.copy(text)
            time.sleep(0.1) # 클립보드 복사 대기
            
            # OS에 따른 단축키 처리 (Windows: ctrl+v, Mac: command+v)
            modifier = "command" if platform.system() == "Darwin" else "ctrl"
            pyautogui.hotkey(modifier, "v")
            
            return f"Typed text via clipboard: {text[:10]}..."
            
        return self._execute_with_wait("type_in_marker", _type)

    def scroll(self, direction: str = "down", clicks: int = 500) -> Dict[str, Any]:
        """화면을 스크롤합니다."""
        def _scroll():
            # 양수는 위로, 음수는 아래로 스크롤 (Windows 기준)
            amount = -clicks if direction == "down" else clicks
            pyautogui.scroll(amount)
            return f"Scrolled {direction} by {clicks} clicks"
            
        return self._execute_with_wait("scroll", _scroll)
        
    def press_key(self, key: str) -> Dict[str, Any]:
        """특정 특수키(Enter, ESC 등)를 누릅니다."""
        def _press():
            pyautogui.press(key)
            return f"Pressed {key}"
            
        return self._execute_with_wait("press_key", _press)

    def finish_task(self, final_data: Any) -> Dict[str, Any]:
        """작업을 완료하고 데이터를 반환합니다."""
        logger.info("Task finished by agent")
        # finish_task는 화면 대기를 할 필요가 없으므로 직접 반환
        return {
            "status": "success",
            "action": "finish_task",
            "result": final_data
        }
