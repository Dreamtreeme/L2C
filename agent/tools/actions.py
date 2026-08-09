import os
import subprocess
from pathlib import Path
import time
from typing import Dict, Any, List
import platform

import pyautogui
import pyperclip
import pygetwindow as gw

from agent.config import get_settings
from agent.sites import get_official_site_url
from agent.utils.logger import logger
from agent.tools.perception import PerceptionEngine


_WINDOWS_WHEEL_DELTA = 120


class ActionTools:
    """
    물리적인 마우스/키보드 조작을 담당하는 Action 도구 모음입니다.
    화면 안정화 대기는 PerceptionEngine.capture_screen()이 담당합니다.
    """

    def __init__(self, perception_engine: PerceptionEngine):
        self.perception = perception_engine
        settings = get_settings().browser
        self.action_pause_sec = settings.action_pause_sec
        self.move_duration_sec = settings.action_move_duration_sec
        self.input_delay_sec = settings.action_input_delay_sec
        self.clipboard_delay_sec = settings.action_clipboard_delay_sec

        # pyautogui 기본 안전 설정
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = self.action_pause_sec

    def _sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def _window_id(self, window: Any) -> int | None:
        return self.perception._window_id(window)

    def _browser_windows(self) -> list[Any]:
        return [
            window
            for window in gw.getAllWindows()
            if self.perception._is_visible_window(window)
            and self.perception._looks_like_browser_window(window)
        ]

    def _browser_window_ids(self) -> set[int]:
        return {
            window_id
            for window_id in (self._window_id(win) for win in self._browser_windows())
            if window_id
        }

    def _bound_browser_window_exists(self) -> bool:
        preferred_id = self.perception.browser_window_id
        if not preferred_id:
            return False
        return preferred_id in self._browser_window_ids()

    def _bind_browser_window(self, window: Any | None) -> bool:
        if not window:
            return False
        return self.perception.bind_browser_window(window)

    def _bind_new_or_active_browser_window(
        self, before_ids: set[int] | None = None
    ) -> bool:
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
        if target:
            self._normalize_browser_window(target)
        return self._bind_browser_window(target)

    def _browser_window_dimensions(self) -> tuple[int, int]:
        settings = get_settings().browser
        return settings.vision_window_width, settings.vision_window_height

    def _browser_profile_dir(self) -> Path:
        """사용자 Chrome과 분리된 자동화 전용 프로필 경로를 준비한다."""

        profile_dir = get_settings().paths.browser_profile_dir
        profile_dir.mkdir(parents=True, exist_ok=True)
        return profile_dir

    def _normalize_browser_window(self, window: Any) -> bool:
        """전용 브라우저의 실제 창 크기를 고정해 비전 좌표계를 안정화한다."""

        width, height = self._browser_window_dimensions()
        if width <= 0 or height <= 0:
            return False
        try:
            if bool(getattr(window, "isMaximized", False)):
                window.restore()
            window.resizeTo(width, height)
            self._sleep(get_settings().browser.resize_wait_sec)
            logger.info(
                "Normalized browser window geometry",
                requested_width=width,
                requested_height=height,
                actual_width=int(getattr(window, "width", 0) or 0),
                actual_height=int(getattr(window, "height", 0) or 0),
            )
            return True
        except Exception as exc:
            logger.warning("Browser window normalization failed", error=str(exc))
            return False

    def _browser_executable(self) -> Path | None:
        configured = str(get_settings().browser.executable or "").strip().strip('"')
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

    def _browser_window_cli_args(self) -> list[str]:
        args = [
            f"--user-data-dir={self._browser_profile_dir()}",
            "--no-first-run",
            "--no-default-browser-check",
            "--hide-crash-restore-bubble",
            "--disable-popup-blocking",
        ]
        width, height = self._browser_window_dimensions()
        if width <= 0 or height <= 0:
            return args
        return [*args, f"--window-size={width},{height}"]

    def _reset_browser_zoom(self) -> None:
        """사이트별로 기억된 브라우저 확대율을 100%로 되돌린다."""

        wait_sec = get_settings().browser.zoom_reset_wait_sec
        self._sleep(wait_sec)
        modifier = "command" if platform.system() == "Darwin" else "ctrl"
        try:
            pyautogui.hotkey(modifier, "0")
        except Exception as exc:
            logger.warning("Browser zoom reset skipped", error=str(exc))

    def _open_url_in_new_window(self, url: str) -> dict[str, Any]:
        before_ids = self._browser_window_ids()
        browser_exe = self._browser_executable()
        launcher = "webbrowser.open_new"
        if browser_exe:
            subprocess.Popen(
                [
                    str(browser_exe),
                    "--new-window",
                    *self._browser_window_cli_args(),
                    url,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            launcher = str(browser_exe)
        else:
            import webbrowser

            webbrowser.open_new(url)

        self._sleep(get_settings().browser.open_wait_sec)
        bound = self._bind_new_or_active_browser_window(before_ids)
        return {
            "opened": True,
            "url": url,
            "reason": "new_browser_window",
            "launcher": launcher,
            "bound_window": bound,
        }

    def _navigate_bound_browser(self, url: str) -> dict[str, Any]:
        region = self.perception._get_browser_region()
        if not region:
            return self._open_url_in_new_window(url)

        modifier = "command" if platform.system() == "Darwin" else "ctrl"
        old_pause = pyautogui.PAUSE
        key_pause = get_settings().vision.url_key_pause_sec
        pyautogui.PAUSE = min(old_pause, key_pause)
        try:
            pyautogui.hotkey(modifier, "l")
            pyperclip.copy(url)
            self._sleep(self.clipboard_delay_sec)
            pyautogui.hotkey(modifier, "v")
            pyautogui.press("enter")
        finally:
            pyautogui.PAUSE = old_pause
        return {"opened": True, "url": url, "reason": "dedicated_browser_navigated"}

    def _action_region(self):
        return self.perception.last_region or self.perception._get_browser_region()

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
        scale_x = self.perception.scale_x
        scale_y = self.perception.scale_y

        x_absolute = int(region["left"] * scale_x) + x_center_relative
        y_absolute = int(region["top"] * scale_y) + y_center_relative

        logger.info(
            f"DPI scaled absolute coords: logical_left={region['left']}, scale_x={scale_x:.2f}, relative_x={x_center_relative} => absolute_x={x_absolute}"
        )
        logger.info(
            f"DPI scaled absolute coords: logical_top={region['top']}, scale_y={scale_y:.2f}, relative_y={y_center_relative} => absolute_y={y_absolute}"
        )

        return x_absolute, y_absolute

    def click_marker(self, bbox: List[int]) -> Dict[str, Any]:
        """마커(UI 요소)의 중앙을 클릭합니다."""

        def _click():
            x, y = self._get_absolute_coords(bbox)
            pyautogui.moveTo(x, y, duration=self.move_duration_sec)
            pyautogui.click()
            return f"Clicked at ({x}, {y})"

        return self._execute("click_marker", _click)

    def type_in_marker(self, bbox: List[int], text: str) -> Dict[str, Any]:
        """마커를 클릭한 후, 기존 텍스트를 지우고 pyperclip을 통해 안전하게 한글/영문 텍스트를 붙여넣습니다."""

        def _type():
            x, y = self._get_absolute_coords(bbox)
            pyautogui.moveTo(x, y, duration=self.move_duration_sec)
            pyautogui.click()
            self._sleep(self.input_delay_sec)

            # OS에 따른 제어 특수키 설정 (Mac: command, Windows: ctrl)
            modifier = "command" if platform.system() == "Darwin" else "ctrl"

            # 기존 입력값을 완전히 지우기 위한 전체선택(Ctrl+A) -> 백스페이스(Backspace) 수행
            pyautogui.hotkey(modifier, "a")
            pyautogui.press("backspace")

            # 클립보드를 통한 한글 씹힘 방지 타이핑
            pyperclip.copy(text)
            self._sleep(self.clipboard_delay_sec)

            pyautogui.hotkey(modifier, "v")

            return f"Typed text via clipboard: {text}"

        return self._execute("type_in_marker", _type)

    def scroll(
        self,
        direction: str = "down",
        bbox: List[int] | None = None,
        amount: str = "page",
    ) -> Dict[str, Any]:
        """전체 페이지 또는 지정한 화면 영역을 물리적으로 스크롤합니다."""

        def _scroll():
            if direction not in {"down", "up", "left", "right"}:
                raise ValueError(f"Unsupported scroll direction: {direction}")
            if amount not in {"small", "page"}:
                raise ValueError(f"Unsupported scroll amount: {amount}")

            if bbox:
                x, y = self._get_absolute_coords(bbox)
                pyautogui.moveTo(
                    x,
                    y,
                    duration=self.move_duration_sec,
                )
            else:
                region = self.perception.last_region
                if region:
                    x = region["left"] + region["width"] // 2
                    y = region["top"] + region["height"] // 2
                    pyautogui.moveTo(x, y, duration=0)
                else:
                    win = gw.getActiveWindow()
                    if win:
                        pyautogui.moveTo(
                            win.left + win.width // 2,
                            win.top + win.height // 2,
                            duration=0,
                        )

            # 전체 페이지의 한 화면 이동은 기존 PageUp/PageDown 동작을 유지합니다.
            if bbox is None and amount == "page" and direction in {"down", "up"}:
                key_to_press = "pagedown" if direction == "down" else "pageup"
                pyautogui.press(key_to_press)
                logger.info(f"Pressed {key_to_press} for scrolling {direction}")
                return f"Scrolled {direction} via {key_to_press}"

            browser_settings = get_settings().browser
            steps = (
                browser_settings.scroll_page_steps
                if amount == "page"
                else browser_settings.scroll_small_steps
            )
            signed_steps = steps if direction in {"up", "left"} else -steps
            # pyautogui 0.9.54의 Windows 구현은 mouse_event에 값을 그대로 넘긴다.
            # Win32 휠 한 칸은 120 단위이므로 실제 칸 수로 환산해야 한다.
            wheel_delta = (
                signed_steps * _WINDOWS_WHEEL_DELTA
                if platform.system() == "Windows"
                else signed_steps
            )
            if direction in {"left", "right"}:
                if platform.system() == "Windows":
                    # PyAutoGUI의 hscroll은 Windows에서 구현되지 않아 Shift+휠을 사용합니다.
                    pyautogui.keyDown("shift")
                    try:
                        pyautogui.scroll(wheel_delta)
                    finally:
                        pyautogui.keyUp("shift")
                    method = "shift+wheel"
                else:
                    pyautogui.hscroll(wheel_delta)
                    method = "horizontal wheel"
            else:
                pyautogui.scroll(wheel_delta)
                method = "vertical wheel"
            logger.info(
                "Scrolled physical region",
                direction=direction,
                amount=amount,
                steps=steps,
                targeted=bool(bbox),
            )
            return f"Scrolled {direction} via {method} ({amount})"

        return self._execute("scroll", _scroll)

    def press_key(self, key: str) -> Dict[str, Any]:
        """특정 특수키(Enter, ESC 등)를 누릅니다."""

        def _press():
            pyautogui.press(key)
            return f"Pressed {key}"

        return self._execute("press_key", _press)

    def open_browser(
        self,
        url: str = "",
        current_url: str = "",
        site: str = "",
    ) -> Dict[str, Any]:
        """요청 사이트의 공식 주소를 전용 브라우저 창에서 연다."""

        target_url = str(url or "").strip()
        if site:
            target_url = get_official_site_url(site)
        if not target_url:
            raise ValueError("url or site is required")

        def _open():
            if self._bound_browser_window_exists():
                return self._navigate_bound_browser(target_url)

            result = self._open_url_in_new_window(target_url)
            self._reset_browser_zoom()
            return result

        logger.info(
            "Opening browser target",
            requested_site=site,
            target_url=target_url,
        )
        return self._execute("open_browser", _open)

    def _find_browser_window(self):
        preferred_id = self.perception.browser_window_id
        if preferred_id:
            for window in self._browser_windows():
                if self._window_id(window) == preferred_id:
                    return window
            logger.info(
                "Bound browser window disappeared; refusing to close another window",
                window_id=preferred_id,
            )
            self.perception.clear_browser_window()
            return None

        active_window = gw.getActiveWindow()
        if (
            active_window
            and self.perception._looks_like_browser_window(active_window)
            and not bool(getattr(active_window, "isMinimized", False))
        ):
            return active_window

        for window in gw.getAllWindows():
            if not self.perception._looks_like_browser_window(window):
                continue
            if getattr(window, "isMinimized", False):
                continue
            if getattr(window, "width", 1) <= 0 or getattr(window, "height", 1) <= 0:
                continue
            return window
        return None

    def close_browser(self) -> Dict[str, Any]:
        """바인딩된 자동화 브라우저 창을 닫습니다."""

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
            self.perception.clear_browser_window()
            return {"closed": True, "title": title}

        return self._execute("close_browser", _close)

    def close_current_tab(self) -> Dict[str, Any]:
        """현재 활성 브라우저 탭 하나만 닫습니다."""

        def _close_tab():
            self.perception._get_browser_region()
            self.perception.release_address_bar_focus(key_pause=0.02)
            modifier = "command" if platform.system() == "Darwin" else "ctrl"
            pyautogui.hotkey(modifier, "w")
            return "Closed current browser tab"

        return self._execute("close_current_tab", _close_tab)

    def switch_tab(self, direction: str) -> Dict[str, Any]:
        """현재 브라우저 창에서 다음 또는 이전 탭으로 전환합니다."""

        def _switch_tab():
            if direction not in {"next", "previous"}:
                raise ValueError(f"Unsupported tab direction: {direction}")
            self.perception._get_browser_region()
            self.perception.release_address_bar_focus(key_pause=0.02)
            modifier = "command" if platform.system() == "Darwin" else "ctrl"
            keys = (
                (modifier, "tab") if direction == "next" else (modifier, "shift", "tab")
            )
            pyautogui.hotkey(*keys)
            return f"Switched to {direction} browser tab"

        return self._execute("switch_tab", _switch_tab)

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
