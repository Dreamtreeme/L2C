import time
from typing import Dict, Any, List
import platform

import pyautogui
import pyperclip
import pygetwindow as gw

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
            import os
            skip_wait = os.getenv("SKIP_WAIT_STABLE", "false").lower() == "true"
            if skip_wait:
                is_stable = True
                logger.info("Bypassing screen stabilization (SKIP_WAIT_STABLE is true)")
            else:
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
        
        # 고해상도 DPI 화면 대응을 위한 논리 좌표 -> 물리 좌표 스케일링 적용
        scale_x = getattr(self.perception, "scale_x", 1.0)
        scale_y = getattr(self.perception, "scale_y", 1.0)
        
        x_absolute = int(region["left"] * scale_x) + x_center_relative
        y_absolute = int(region["top"] * scale_y) + y_center_relative
        
        logger.info(f"DPI scaled absolute coords: logical_left={region['left']}, scale_x={scale_x:.2f}, relative_x={x_center_relative} => absolute_x={x_absolute}")
        logger.info(f"DPI scaled absolute coords: logical_top={region['top']}, scale_y={scale_y:.2f}, relative_y={y_center_relative} => absolute_y={y_absolute}")
        
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
        """마커를 클릭한 후, 기존 텍스트를 지우고 pyperclip을 통해 안전하게 한글/영문 텍스트를 붙여넣습니다."""
        def _type():
            x, y = self._get_absolute_coords(bbox)
            pyautogui.moveTo(x, y, duration=0.2)
            pyautogui.click()
            time.sleep(0.1)
            
            # OS에 따른 제어 특수키 설정 (Mac: command, Windows: ctrl)
            modifier = "command" if platform.system() == "Darwin" else "ctrl"
            
            # 기존 입력값을 완전히 지우기 위한 전체선택(Ctrl+A) -> 백스페이스(Backspace) 수행
            pyautogui.hotkey(modifier, "a")
            time.sleep(0.05)
            pyautogui.press("backspace")
            time.sleep(0.05)
            
            # 클립보드를 통한 한글 씹힘 방지 타이핑
            pyperclip.copy(text)
            time.sleep(0.1) # 클립보드 복사 대기
            
            pyautogui.hotkey(modifier, "v")
            time.sleep(0.1)
            
            return f"Typed text via clipboard: {text}"
            
        return self._execute_with_wait("type_in_marker", _type)

    def scroll(self, direction: str = "down") -> Dict[str, Any]:
        """화면을 스크롤합니다."""
        def _scroll():
            # 활성 창(브라우저)의 중앙을 클릭하여 포커스 확보
            win = gw.getActiveWindow()
            if win:
                pyautogui.click(win.left + win.width // 2, win.top + win.height // 2)
                time.sleep(0.1)
                
            key_to_press = "pagedown" if direction == "down" else "pageup"
            pyautogui.press(key_to_press)
            logger.info(f"Pressed {key_to_press} for scrolling {direction}")
            
            return f"Scrolled {direction} via {key_to_press}"
            
        return self._execute_with_wait("scroll", _scroll)
        
    def press_key(self, key: str) -> Dict[str, Any]:
        """특정 특수키(Enter, ESC 등)를 누릅니다."""
        def _press():
            pyautogui.press(key)
            return f"Pressed {key}"
            
        return self._execute_with_wait("press_key", _press)

    def open_browser(self, url: str) -> Dict[str, Any]:
        """기본 브라우저를 열고 지정된 URL로 이동합니다."""
        def _open():
            import webbrowser
            webbrowser.open(url)
            # 브라우저 창이 뜨고 페이지가 로딩될 시간을 조금 줍니다.
            time.sleep(3)
            return f"Opened browser with URL: {url}"
            
        return self._execute_with_wait("open_browser", _open)
        
    def get_credentials(self, site: str) -> Dict[str, Any]:
        """지정된 사이트의 자격 증명(ID/PW)을 가져옵니다."""
        def _get():
            from agent.credentials.manager import CredentialManager
            cm = CredentialManager()
            creds = cm.get_credentials(site)
            if creds and creds[0] and creds[1]:
                return {"username": creds[0], "password": creds[1]}
            else:
                raise ValueError(f"No credentials found for site: {site}")
                
        # 화면 대기(WaitStable)가 필요 없는 정보 조회 액션
        logger.info(f"Executing action: get_credentials for {site}")
        try:
            result = _get()
            return {
                "status": "success",
                "action": "get_credentials",
                "result": result
            }
        except Exception as e:
            logger.exception("Failed to get credentials", error=str(e))
            return {
                "status": "error",
                "action": "get_credentials",
                "error": str(e)
            }

    def go_back(self) -> Dict[str, Any]:
        """브라우저의 뒤로가기 동작을 수행합니다 (Alt + Left Arrow)."""
        def _back():
            pyautogui.hotkey('alt', 'left')
            return "Navigated back using Alt + Left Arrow shortcut"
            
        return self._execute_with_wait("go_back", _back)

    def finish_task(self, final_data: Any) -> Dict[str, Any]:
        """작업을 완료하고 데이터를 반환합니다."""
        logger.info("Task finished by agent")
        # finish_task는 화면 대기를 할 필요가 없으므로 직접 반환
        return {
            "status": "success",
            "action": "finish_task",
            "result": final_data
        }
