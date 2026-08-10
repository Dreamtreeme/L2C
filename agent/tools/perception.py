import ctypes
import ctypes.wintypes as wintypes
import datetime
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mss
import mss.tools
import pyautogui
import pygetwindow as gw
import pyperclip
from PIL import Image

from agent.config import get_settings
from agent.tools.ocr_engine import OcrEngine
from agent.utils.logger import logger
from agent.vision.loading_wait import LoadingWait


class BrowserWindowNotFoundError(RuntimeError):
    """캡처할 브라우저 창을 현재 데스크톱에서 찾지 못한 경우."""


class PerceptionEngine:
    """
    모니터 화면을 인식하고 분석하는 Perception 엔진입니다.
    mss 화면 캡처와 OCR 진입점을 담당합니다.
    """

    def __init__(self):
        self.screenshot_dir = get_settings().paths.screenshot_dir
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.sct = mss.mss()
        self.ocr_engine = OcrEngine()
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.last_region = None
        self._browser_window_id = None
        self.last_capture_quality: Dict[str, Any] = {}

        self.loading_wait = LoadingWait(self)

        logger.info(
            "Perception engine initialized",
            screenshot_dir=str(self.screenshot_dir),
        )

    def close(self) -> None:
        """OCR 하위 프로세스와 화면 캡처 핸들을 명시적으로 정리한다."""

        self.ocr_engine.close()
        self.sct.close()
        self.clear_browser_window()

    @property
    def browser_window_id(self) -> int | None:
        return self._browser_window_id

    @property
    def ocr_worker_pid(self) -> int | None:
        return self.ocr_engine.worker_pid

    def ensure_ocr_worker_ready(self) -> None:
        self.ocr_engine.ensure_ready()

    @staticmethod
    def _window_id(window) -> int | None:
        return getattr(window, "_hWnd", None) or getattr(window, "hWnd", None)

    @staticmethod
    def _is_visible_window(window) -> bool:
        return (
            bool(getattr(window, "visible", True))
            and not bool(getattr(window, "isMinimized", False))
            and int(getattr(window, "width", 0) or 0) > 0
            and int(getattr(window, "height", 0) or 0) > 0
        )

    @staticmethod
    def _looks_like_browser_window(window) -> bool:
        title = str(getattr(window, "title", "") or "")
        if not title:
            return False
        lowered = title.lower()
        keywords = (
            "chrome",
            "chromium",
            "microsoft edge",
            "firefox",
            "brave",
            "whale",
            "네이버 웨일",
            "웨일",
            "크롬",
        )
        return any(keyword in lowered for keyword in keywords)

    def bind_browser_window(self, window) -> bool:
        window_id = self._window_id(window)
        if not window_id:
            return False
        self._browser_window_id = window_id
        logger.info(
            "Bound perception to browser window",
            window_id=window_id,
            title=str(getattr(window, "title", "") or "")[:120],
        )
        return True

    def clear_browser_window(self) -> None:
        self._browser_window_id = None
        self.last_region = None

    def _browser_region_from_window(self, win) -> Dict[str, int]:
        try:
            win.activate()
        except Exception as e:
            logger.debug("Failed to activate window (bring to front)", error=str(e))

        region = self._client_region(win) or {
            "top": int(win.top),
            "left": int(win.left),
            "width": int(win.width),
            "height": int(win.height),
        }
        work_area = self._monitor_work_area(win)
        return self._intersect_regions(region, work_area) if work_area else region

    @classmethod
    def _client_region(cls, window) -> Optional[Dict[str, int]]:
        """Windows 창의 고정 테두리를 제외한 실제 클라이언트 영역을 읽는다."""

        if os.name != "nt":
            return None
        window_id = cls._window_id(window)
        if not window_id:
            return None
        try:
            rect = wintypes.RECT()
            origin = wintypes.POINT(0, 0)
            user32 = ctypes.windll.user32
            if not user32.GetClientRect(wintypes.HWND(window_id), ctypes.byref(rect)):
                return None
            if not user32.ClientToScreen(
                wintypes.HWND(window_id),
                ctypes.byref(origin),
            ):
                return None
            return {
                "top": int(origin.y),
                "left": int(origin.x),
                "width": int(rect.right - rect.left),
                "height": int(rect.bottom - rect.top),
            }
        except Exception as exc:
            logger.debug("Browser client region lookup failed", error=str(exc))
            return None

    @staticmethod
    def _intersect_regions(
        region: Dict[str, int],
        bounds: Dict[str, int],
    ) -> Dict[str, int]:
        """화면 캡처 영역을 모니터 작업영역 안으로 제한한다."""

        left = max(int(region["left"]), int(bounds["left"]))
        top = max(int(region["top"]), int(bounds["top"]))
        right = min(
            int(region["left"]) + int(region["width"]),
            int(bounds["left"]) + int(bounds["width"]),
        )
        bottom = min(
            int(region["top"]) + int(region["height"]),
            int(bounds["top"]) + int(bounds["height"]),
        )
        if right <= left or bottom <= top:
            return dict(region)
        return {
            "top": top,
            "left": left,
            "width": right - left,
            "height": bottom - top,
        }

    @classmethod
    def _monitor_work_area(cls, window) -> Optional[Dict[str, int]]:
        """현재 창이 있는 Windows 모니터의 작업표시줄 제외 영역을 반환한다."""

        if os.name != "nt":
            return None
        window_id = cls._window_id(window)
        if not window_id:
            return None
        try:

            class MonitorInfo(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT),
                    ("dwFlags", wintypes.DWORD),
                ]

            user32 = ctypes.windll.user32
            monitor = user32.MonitorFromWindow(wintypes.HWND(window_id), 2)
            if not monitor:
                return None
            info = MonitorInfo()
            info.cbSize = ctypes.sizeof(MonitorInfo)
            if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                return None
            return {
                "top": int(info.rcWork.top),
                "left": int(info.rcWork.left),
                "width": int(info.rcWork.right - info.rcWork.left),
                "height": int(info.rcWork.bottom - info.rcWork.top),
            }
        except Exception as exc:
            logger.debug("Monitor work area lookup failed", error=str(exc))
            return None

    def _find_browser_window(self):
        windows = [win for win in gw.getAllWindows() if self._is_visible_window(win)]
        preferred_id = self._browser_window_id
        if preferred_id:
            for win in windows:
                if self._window_id(
                    win
                ) == preferred_id and self._looks_like_browser_window(win):
                    return win
            logger.info(
                "Preferred browser window disappeared; clearing binding",
                window_id=preferred_id,
            )
            self.clear_browser_window()
            return None

        active = gw.getActiveWindow()
        if (
            active
            and self._is_visible_window(active)
            and self._looks_like_browser_window(active)
        ):
            self.bind_browser_window(active)
            return active

        return None

    def _get_browser_region(self) -> Optional[Dict[str, int]]:
        """바인딩된 창 또는 현재 활성 브라우저의 캡처 영역을 반환한다."""
        win = self._find_browser_window()
        if not win:
            return None
        return self._browser_region_from_window(win)

    def capture_screen(
        self,
        filename: Optional[str] = None,
    ) -> Path:
        """현재 브라우저 화면을 즉시 한 장 저장한다."""

        region = self._require_browser_region()
        return self._save_capture(region, filename)

    def _require_browser_region(self) -> Dict[str, int]:
        region = self._get_browser_region()
        if region is None:
            raise BrowserWindowNotFoundError("캡처할 브라우저 창을 찾지 못했습니다.")
        return region

    def _save_capture(
        self,
        region: Dict[str, int],
        filename: Optional[str] = None,
    ) -> Path:
        """이미 찾은 브라우저 영역을 다시 조회하지 않고 저장한다."""

        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screen_{timestamp}.png"

        output_path = self.screenshot_dir / filename

        try:
            sct_img = self.sct.grab(region)
            self.scale_x = sct_img.width / region["width"]
            self.scale_y = sct_img.height / region["height"]
            self.last_region = region
            logger.debug(
                "Captured browser window only",
                region=region,
                scale_x=self.scale_x,
                scale_y=self.scale_y,
            )

            # Convert to PIL Image and preserve text edges by default.
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            if output_path.suffix.lower() in (".jpg", ".jpeg"):
                img.save(str(output_path), "JPEG", quality=80)
            else:
                img.save(str(output_path))

            logger.info(
                "Screen captured successfully",
                width=sct_img.width,
                height=sct_img.height,
                output_path=str(output_path),
            )
            return output_path
        except Exception as e:
            logger.exception("Failed to capture screen", error=str(e))
            raise

    def wait_for_transition_change(self, reference_image_path: str) -> bool:
        """화면 변경 행동 뒤 이전 화면이 그대로인 동안에는 OCR 캡처를 미룹니다."""
        region = self._require_browser_region()
        return self.loading_wait.wait_for_change(
            reference_image_path,
            region=region,
        )

    def capture_usable_screen(
        self,
        *,
        initial_wait_sec: Optional[float] = None,
    ) -> Path:
        """메모리 프레임으로 로딩을 기다린 뒤 준비된 화면만 한 번 저장한다."""

        wait_sec = (
            get_settings().vision.capture_initial_wait_sec
            if initial_wait_sec is None
            else initial_wait_sec
        )
        if wait_sec > 0:
            time.sleep(wait_sec)
        region = self._require_browser_region()
        self.last_capture_quality = self.loading_wait.wait_until_ready(region=region)
        return self._save_capture(region)

    def get_current_url(self) -> str:
        """활성 브라우저의 주소창 URL을 클립보드 경유로 읽습니다."""
        url = ""
        try:
            old_pause = pyautogui.PAUSE
            modifier = "command" if platform.system() == "Darwin" else "ctrl"
            settings = get_settings().vision
            key_pause = settings.url_key_pause_sec
            copy_wait = settings.url_copy_wait_sec
            copy_timeout = settings.url_copy_timeout_sec
            copy_attempts = settings.url_copy_attempts
            pyautogui.PAUSE = min(old_pause, key_pause)
            try:
                url = self._copy_address_bar_url(
                    pyautogui,
                    pyperclip,
                    modifier=modifier,
                    key_pause=key_pause,
                    copy_wait=copy_wait,
                    copy_timeout=copy_timeout,
                    max_attempts=copy_attempts,
                )
            finally:
                self.release_address_bar_focus(key_pause=key_pause)
                pyautogui.PAUSE = old_pause
            if url.startswith(("http://", "https://")):
                return url
        except Exception as e:
            logger.debug("Failed to read current browser URL", error=str(e))
        return ""

    def _copy_address_bar_url(
        self,
        pyautogui_module,
        pyperclip_module,
        *,
        modifier: str,
        key_pause: float,
        copy_wait: float,
        copy_timeout: float,
        max_attempts: int,
    ) -> str:
        """주소창 선택이나 클립보드 반영이 늦을 때 동일한 물리 입력을 제한적으로 재시도한다."""
        max_attempts = max(1, int(max_attempts))
        for attempt in range(1, max_attempts + 1):
            # 새 탭 전환 직후에도 입력 대상이 브라우저임을 다시 보장한다.
            self._get_browser_region()
            pyperclip_module.copy("")
            if key_pause > 0:
                time.sleep(key_pause)
            pyautogui_module.hotkey(modifier, "l")
            if key_pause > 0:
                time.sleep(key_pause)
            pyautogui_module.hotkey(modifier, "c")

            deadline = time.monotonic() + max(0.0, copy_timeout)
            while True:
                url = (pyperclip_module.paste() or "").strip()
                if url.startswith(("http://", "https://")):
                    if attempt > 1:
                        logger.info("Browser URL copy recovered", attempt=attempt)
                    return url
                if time.monotonic() >= deadline:
                    break
                time.sleep(max(0.005, copy_wait))

            logger.debug(
                "Browser URL copy attempt failed",
                attempt=attempt,
                max_attempts=max_attempts,
            )
        return ""

    def release_address_bar_focus(self, key_pause: float = 0.02) -> None:
        """주소창 URL 복사 후 남는 브라우저 툴바 포커스를 페이지 쪽으로 되돌립니다."""
        old_pause = pyautogui.PAUSE
        pyautogui.PAUSE = min(old_pause, key_pause)
        try:
            pyautogui.press("esc")
            pyautogui.hotkey("ctrl", "f6")
        finally:
            pyautogui.PAUSE = old_pause

    def analyze_ui(self, image_path: Path) -> Dict[str, Any]:
        """PaddleOCR와 OmniParser 결과를 물리 좌표가 있는 마커로 변환한다."""

        if not image_path.exists():
            logger.error(
                "Image file not found for UI analysis", image_path=str(image_path)
            )
            raise FileNotFoundError(f"Image not found: {image_path}")

        crop_top = int(self.last_capture_quality.get("content_top", 0) or 0)
        marked_filename = f"marked_{image_path.name}"
        marked_path, marker_bboxes, final_elements = self.ocr_engine.process_image(
            image_path,
            output_filename=marked_filename,
            content_top=crop_top,
        )

        markers = []
        for marker_id, bbox in marker_bboxes.items():
            elem = final_elements[marker_id] if marker_id < len(final_elements) else {}
            elem_type = elem.get("type", "element")
            local_text = str(elem.get("text") or "")
            text = (
                local_text
                if elem_type == "text" and local_text
                else f"상호작용 가능한 요소 ({elem_type})"
            )
            marker = {
                "id": marker_id,
                "text": text,
                "bbox": bbox,
                "type": elem.get("type", "element"),
            }
            if elem.get("conf") is not None:
                marker["conf"] = elem.get("conf")
            markers.append(marker)

        logger.info("UI analysis pipeline complete", final_markers_count=len(markers))
        return {
            "markers": markers,
            "marked_image": str(marked_path),
            "content_top": crop_top,
        }
