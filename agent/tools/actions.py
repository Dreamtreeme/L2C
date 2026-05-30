import os
from urllib.parse import urlparse
import time
from typing import Dict, Any, List
import platform

import pyautogui
import pyperclip
import pygetwindow as gw

from agent.utils.logger import logger
from agent.tools.perception import PerceptionEngine

class ActionTools:
    """
    물리적인 마우스/키보드 조작을 담당하는 Action 도구 모음입니다.
    화면 안정화 대기는 PerceptionEngine.capture_screen()이 담당합니다.
    """

    def __init__(self, perception_engine: PerceptionEngine):
        self.perception = perception_engine
        self.action_pause_sec = self._env_float("VISION_ACTION_PAUSE_SEC", 0.03)
        self.move_duration_sec = self._env_float("VISION_ACTION_MOVE_DURATION_SEC", 0.05)
        self.input_delay_sec = self._env_float("VISION_ACTION_INPUT_DELAY_SEC", 0.02)
        self.clipboard_delay_sec = self._env_float("VISION_ACTION_CLIPBOARD_DELAY_SEC", 0.02)

        # pyautogui 기본 안전 설정
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = self.action_pause_sec

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return max(0.0, float(os.getenv(name, str(default))))
        except ValueError:
            return default

    def _cfg_float(self, attr: str, env_name: str, default: float) -> float:
        return getattr(self, attr, self._env_float(env_name, default))

    def _sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def _action_region(self):
        return getattr(self.perception, "last_region", None) or self.perception._get_browser_region()

    @staticmethod
    def _same_site_or_url(left: str, right: str) -> bool:
        if not left or not right:
            return False
        left_parsed = urlparse(left)
        right_parsed = urlparse(right)
        if not left_parsed.netloc or not right_parsed.netloc:
            return False
        left_host = left_parsed.netloc.lower().removeprefix("www.")
        right_host = right_parsed.netloc.lower().removeprefix("www.")
        if left_host != right_host:
            return False
        right_path = right_parsed.path.rstrip("/")
        if right_path in ("", "/"):
            return True
        return left_parsed.path.rstrip("/") == right_path

    def _execute(self, action_name: str, func, *args, **kwargs) -> Dict[str, Any]:
        """
        액션을 실행하고 결과를 반환합니다.
        화면 안정화 대기는 다음 perception_node의 capture_screen()이 처리합니다.
        """
        logger.info(f"Executing action: {action_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Action '{action_name}' completed")
            return {"status": "success", "action": action_name, "result": result}
        except Exception as e:
            logger.exception(f"Action '{action_name}' failed", error=str(e))
            return {"status": "error", "action": action_name, "error": str(e)}
            
    def _get_absolute_coords(self, bbox: List[int]) -> tuple[int, int]:
        """
        상대적인 bbox 좌표를 현재 브라우저 영역의 절대 좌표로 변환하고
        해당 박스의 정중앙을 반환합니다.
        bbox: [x_min, y_min, x_max, y_max] (브라우저 창 내부 상대 좌표라고 가정)
        """
        region = self._action_region()
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
            pyautogui.moveTo(x, y, duration=self._cfg_float("move_duration_sec", "VISION_ACTION_MOVE_DURATION_SEC", 0.05))
            pyautogui.click()
            return f"Clicked at ({x}, {y})"

        return self._execute("click_marker", _click)
        
    def type_in_marker(self, bbox: List[int], text: str) -> Dict[str, Any]:
        """마커를 클릭한 후, 기존 텍스트를 지우고 pyperclip을 통해 안전하게 한글/영문 텍스트를 붙여넣습니다."""
        def _type():
            x, y = self._get_absolute_coords(bbox)
            pyautogui.moveTo(x, y, duration=self._cfg_float("move_duration_sec", "VISION_ACTION_MOVE_DURATION_SEC", 0.05))
            pyautogui.click()
            self._sleep(self._cfg_float("input_delay_sec", "VISION_ACTION_INPUT_DELAY_SEC", 0.02))
            
            # OS에 따른 제어 특수키 설정 (Mac: command, Windows: ctrl)
            modifier = "command" if platform.system() == "Darwin" else "ctrl"
            
            # 기존 입력값을 완전히 지우기 위한 전체선택(Ctrl+A) -> 백스페이스(Backspace) 수행
            pyautogui.hotkey(modifier, "a")
            pyautogui.press("backspace")
            
            # 클립보드를 통한 한글 씹힘 방지 타이핑
            pyperclip.copy(text)
            self._sleep(self._cfg_float("clipboard_delay_sec", "VISION_ACTION_CLIPBOARD_DELAY_SEC", 0.02))
            
            pyautogui.hotkey(modifier, "v")
            
            return f"Typed text via clipboard: {text}"
            
        return self._execute("type_in_marker", _type)

    def scroll(self, direction: str = "down") -> Dict[str, Any]:
        """화면을 스크롤합니다."""
        def _scroll():
            # 활성 창(브라우저)의 중앙을 클릭하여 포커스 확보
            region = getattr(self.perception, "last_region", None)
            if region:
                pyautogui.click(region["left"] + region["width"] // 2, region["top"] + region["height"] // 2)
            else:
                win = gw.getActiveWindow()
                if win:
                    pyautogui.click(win.left + win.width // 2, win.top + win.height // 2)
                
            key_to_press = "pagedown" if direction == "down" else "pageup"
            pyautogui.press(key_to_press)
            logger.info(f"Pressed {key_to_press} for scrolling {direction}")
            
            return f"Scrolled {direction} via {key_to_press}"
            
        return self._execute("scroll", _scroll)
        
    def press_key(self, key: str) -> Dict[str, Any]:
        """특정 특수키(Enter, ESC 등)를 누릅니다."""
        def _press():
            pyautogui.press(key)
            return f"Pressed {key}"
            
        return self._execute("press_key", _press)

    def open_browser(self, url: str, current_url: str = "") -> Dict[str, Any]:
        """기본 브라우저를 열거나, 이미 같은 사이트가 열려 있으면 재사용합니다."""
        def _open():
            if self._same_site_or_url(current_url, url):
                return {"opened": False, "url": current_url, "reason": "state_url_already_matches"}

            if self.perception._get_browser_region():
                browser_url = self.perception.get_current_url()
                if self._same_site_or_url(browser_url, url):
                    return {"opened": False, "url": browser_url, "reason": "active_browser_already_matches"}

            import webbrowser
            webbrowser.open(url)
            return {"opened": True, "url": url}

        return self._execute("open_browser", _open)

    def go_back(self) -> Dict[str, Any]:
        """브라우저의 뒤로가기 동작을 수행합니다."""
        def _back():
            self.perception._get_browser_region()
            self.perception.release_address_bar_focus(key_pause=0.02)
            logger.info("Sending browserback key after releasing address bar focus")
            pyautogui.press("browserback")
            return "Navigated back using browser back key"
            
        return self._execute("go_back", _back)

    def finish_task(self, final_data: Any) -> Dict[str, Any]:
        """작업을 완료하고 데이터를 반환합니다."""
        logger.info("Task finished by agent")
        return self._execute("finish_task", lambda: final_data)
