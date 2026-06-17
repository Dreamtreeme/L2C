import os
import subprocess
from pathlib import Path
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

    def _window_id(self, window: Any) -> int | None:
        helper = getattr(self.perception, "_window_id", None)
        if helper:
            return helper(window)
        return getattr(window, "_hWnd", None) or getattr(window, "hWnd", None)

    def _browser_windows(self) -> list[Any]:
        looks_like = getattr(self.perception, "_looks_like_browser_window", self._looks_like_browser_window)
        is_visible = getattr(
            self.perception,
            "_is_visible_window",
            lambda win: not bool(getattr(win, "isMinimized", False))
            and int(getattr(win, "width", 0) or 0) > 0
            and int(getattr(win, "height", 0) or 0) > 0,
        )
        return [win for win in gw.getAllWindows() if is_visible(win) and looks_like(win)]

    def _browser_window_ids(self) -> set[int]:
        return {window_id for window_id in (self._window_id(win) for win in self._browser_windows()) if window_id}

    def _bound_browser_window_exists(self) -> bool:
        preferred_id = getattr(self.perception, "_browser_window_id", None)
        if not preferred_id:
            return False
        return preferred_id in self._browser_window_ids()

    def _bind_browser_window(self, window: Any | None) -> bool:
        if not window:
            return False
        binder = getattr(self.perception, "bind_browser_window", None)
        if binder:
            return bool(binder(window))
        return False

    def _bind_new_or_active_browser_window(self, before_ids: set[int] | None = None) -> bool:
        before_ids = before_ids or set()
        windows = self._browser_windows()
        active = gw.getActiveWindow()
        active_id = self._window_id(active) if active else None
        new_windows = [win for win in windows if self._window_id(win) not in before_ids]

        target = None
        if active_id and any(self._window_id(win) == active_id for win in new_windows):
            target = active
        elif new_windows:
            target = new_windows[-1]
        elif active and any(self._window_id(win) == active_id for win in windows):
            target = active
        elif windows:
            target = windows[0]
        return self._bind_browser_window(target)

    def _browser_executable(self) -> Path | None:
        configured = os.getenv("VISION_BROWSER_EXECUTABLE", "").strip().strip('"')
        if configured:
            path = Path(configured)
            if path.exists():
                return path

        candidates = []
        for env_name in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            base = os.getenv(env_name)
            if not base:
                continue
            base_path = Path(base)
            candidates.extend(
                [
                    base_path / "Google" / "Chrome" / "Application" / "chrome.exe",
                    base_path / "Microsoft" / "Edge" / "Application" / "msedge.exe",
                ]
            )
        for path in candidates:
            if path.exists():
                return path
        return None

    def _open_url_in_new_window(self, url: str) -> dict[str, Any]:
        before_ids = self._browser_window_ids()
        browser_exe = self._browser_executable()
        launcher = "webbrowser.open_new"
        if browser_exe:
            subprocess.Popen(
                [str(browser_exe), "--new-window", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            launcher = str(browser_exe)
        else:
            import webbrowser
            webbrowser.open_new(url)

        self._sleep(self._env_float("VISION_BROWSER_OPEN_WAIT_SEC", 0.8))
        bound = self._bind_new_or_active_browser_window(before_ids)
        return {"opened": True, "url": url, "reason": "new_browser_window", "launcher": launcher, "bound_window": bound}

    def _navigate_bound_browser(self, url: str) -> dict[str, Any]:
        region = self.perception._get_browser_region()
        if not region:
            return self._open_url_in_new_window(url)

        modifier = "command" if platform.system() == "Darwin" else "ctrl"
        old_pause = pyautogui.PAUSE
        key_pause = self._env_float("VISION_URL_KEY_PAUSE_SEC", 0.015)
        pyautogui.PAUSE = min(old_pause, key_pause)
        try:
            pyautogui.hotkey(modifier, "l")
            pyperclip.copy(url)
            self._sleep(self._cfg_float("clipboard_delay_sec", "VISION_ACTION_CLIPBOARD_DELAY_SEC", 0.02))
            pyautogui.hotkey(modifier, "v")
            pyautogui.press("enter")
        finally:
            pyautogui.PAUSE = old_pause
        return {"opened": True, "url": url, "reason": "dedicated_browser_navigated"}
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
        left_path = left_parsed.path.rstrip("/") or "/"
        right_path = right_parsed.path.rstrip("/") or "/"
        if right_path == "/":
            return left_path in {"/", "/search"}
        if left_path != right_path:
            return False
        if right_parsed.query:
            return left_parsed.query == right_parsed.query
        return True

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
        """Open the first target in a dedicated browser window, then navigate only that bound window."""
        def _open():
            if self._same_site_or_url(current_url, url):
                return {"opened": False, "url": current_url, "reason": "state_url_already_matches"}

            if self._bound_browser_window_exists():
                return self._navigate_bound_browser(url)

            return self._open_url_in_new_window(url)

        return self._execute("open_browser", _open)

    @staticmethod
    def _looks_like_browser_window(window: Any) -> bool:
        return PerceptionEngine._looks_like_browser_window(window)

    def _find_browser_window(self):
        active_window = gw.getActiveWindow()
        if active_window and self._looks_like_browser_window(active_window):
            return active_window

        for window in gw.getAllWindows():
            if not self._looks_like_browser_window(window):
                continue
            if getattr(window, "isMinimized", False):
                continue
            if getattr(window, "width", 1) <= 0 or getattr(window, "height", 1) <= 0:
                continue
            return window
        return None

    def close_browser(self) -> Dict[str, Any]:
        """열려 있는 브라우저 창을 닫습니다."""
        def _close():
            window = self._find_browser_window()
            if not window:
                return {"closed": False, "reason": "browser_not_found"}

            title = str(getattr(window, "title", "") or "")
            try:
                if getattr(window, "isMinimized", False):
                    window.restore()
                window.activate()
            except Exception as e:
                logger.debug(f"Browser window activation skipped before close: {e}")

            window.close()
            if hasattr(self, "perception") and hasattr(self.perception, "clear_browser_window"):
                self.perception.clear_browser_window()
            return {"closed": True, "title": title}

        return self._execute("close_browser", _close)

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
