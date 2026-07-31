import ctypes
import datetime
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mss
import mss.tools
import pygetwindow as gw
from PIL import Image, ImageFilter, ImageStat

from agent.config import get_settings
from agent.utils.logger import logger
from shared.config import SCREENSHOT_DIR


from agent.tools.som_engine import SomEngine


class PerceptionEngine:
    """
    모니터 화면을 인식하고 분석하는 Perception 엔진입니다.
    mss를 이용한 고속 화면 캡처 및 OmniParser 연동을 담당합니다.
    """

    def __init__(self):
        self.screenshot_dir = SCREENSHOT_DIR
        self.sct = mss.mss()
        self.som_engine = SomEngine()
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.last_region = None
        self._browser_window_id = None
        self._last_url = ""
        self.last_capture_quality: Dict[str, Any] = {}
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
        self._analysis_cache_order: list[str] = []
        self._analysis_cache_limit = get_settings().vision.ui_analysis_cache_limit

        # WaitStable은 PerceptionEngine을 역참조하므로 순환 import를 피하기 위해 lazy 로딩합니다.
        from agent.utils.wait_stable import WaitStable
        self._wait_stable = WaitStable(self)

        logger.info("PerceptionEngine initialized with SomEngine", screenshot_dir=str(self.screenshot_dir))

    def close(self) -> None:
        """OCR 하위 프로세스와 화면 캡처 핸들을 명시적으로 정리한다."""

        close_som = getattr(self.som_engine, "close", None)
        if callable(close_som):
            close_som()
        close_capture = getattr(self.sct, "close", None)
        if callable(close_capture):
            close_capture()
        self.clear_browser_window()

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

        border = 8
        top = win.top + border if win.isMaximized else win.top
        region = {
            "top": top,
            "left": win.left + border,
            "width": win.width - (border * 2),
            "height": win.height - border - (border if win.isMaximized else 0),
        }
        work_area = self._monitor_work_area(win)
        return self._intersect_regions(region, work_area) if work_area else region

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
            from ctypes import wintypes

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
                if self._window_id(win) == preferred_id and self._looks_like_browser_window(win):
                    return win
            logger.info("Preferred browser window disappeared; clearing binding", window_id=preferred_id)
            self.clear_browser_window()

        active = gw.getActiveWindow()
        if active and self._is_visible_window(active) and self._looks_like_browser_window(active):
            self.bind_browser_window(active)
            return active

        for win in windows:
            if self._looks_like_browser_window(win):
                self.bind_browser_window(win)
                return win
        return None

    def _get_browser_region(self) -> Optional[Dict[str, int]]:
        """
        Return the bound browser window region when available.
        If no browser is bound yet, bind the active visible browser first, then fall back to any visible browser.
        """
        win = self._find_browser_window()
        if not win:
            return None
        return self._browser_region_from_window(win)

    def capture_screen(
        self,
        filename: Optional[str] = None,
        initial_wait_sec: Optional[float] = None,
        *,
        wait_for_stable: bool = True,
    ) -> Path:
        """
        큰 화면 흔들림이 잦아든 뒤 브라우저 창 영역을 캡처합니다.
        행동 후 변화 시작과 안정화 시간은 OpenCV 프레임 비교로 판단합니다.

        Args:
            filename: 저장할 파일명. 입력하지 않으면 타임스탬프 기반 자동 생성.

        Returns:
            저장된 스크린샷 이미지의 절대 경로 (Path 객체)
        """
        # 액션 효과가 캡처에 반영되기 시작할 짧은 시간만 확보합니다.
        if initial_wait_sec is None:
            initial_wait_sec = get_settings().vision.capture_initial_wait_sec
        if initial_wait_sec > 0:
            time.sleep(initial_wait_sec)

        # Browser region lookup activates the window and is relatively expensive.
        # Resolve it once and share it with WaitStable and the final capture.
        region = self._get_browser_region()
        if wait_for_stable:
            self._wait_stable.wait(region=region)

        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screen_{timestamp}.png"
            
        output_path = self.screenshot_dir / filename
        
        try:
            if region:
                # 브라우저만 캡처
                sct_img = self.sct.grab(region)
                self.scale_x = sct_img.width / region["width"]
                self.scale_y = sct_img.height / region["height"]
                self.last_region = region
                logger.debug("Captured browser window only", region=region, scale_x=self.scale_x, scale_y=self.scale_y)
            else:
                # 브라우저를 못 찾으면 모니터 1번 (주 모니터) 전체 캡처
                monitor = self.sct.monitors[1]
                sct_img = self.sct.grab(monitor)
                self.scale_x = 1.0
                self.scale_y = 1.0
                self.last_region = {
                    "top": monitor["top"],
                    "left": monitor["left"],
                    "width": monitor["width"],
                    "height": monitor["height"]
                }
                logger.debug("Browser not found, captured full monitor", monitor=monitor)
                
            # Convert to PIL Image and preserve text edges by default.
            from PIL import Image
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            if output_path.suffix.lower() in (".jpg", ".jpeg"):
                img.save(str(output_path), "JPEG", quality=80)
            else:
                img.save(str(output_path))
            
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

    def wait_for_transition_change(self, reference_image_path: str) -> bool:
        """화면 변경 행동 뒤 이전 화면이 그대로인 동안에는 OCR 캡처를 미룹니다."""
        region = self._get_browser_region()
        return self._wait_stable.wait_for_change(
            reference_image_path,
            region=region,
        )

    def wait_for_transition_phash_match(
        self,
        target_phash: str,
        *,
        max_distance: int,
        max_wait_sec: float | None = None,
    ) -> bool:
        """목표 화면 pHash가 나타날 때까지 메모리 캡처만 반복합니다."""

        region = self._get_browser_region()
        return self._wait_stable.wait_for_phash_match(
            target_phash,
            max_distance=max_distance,
            max_wait_sec=max_wait_sec,
            region=region,
        )

    def screen_quality(self, image_path: Path) -> Dict[str, Any]:
        """브라우저 본문이 단색 빈 화면에 가까운지 저비용 이미지 지표로 검사한다."""
        settings = get_settings().vision
        bottom_ignore = settings.page_content_bottom_ignore_px
        sample_width = settings.page_quality_sample_width
        try:
            with Image.open(image_path) as source:
                source_rgb = source.convert("RGB")
                content_top_setting = settings.page_content_top_px.strip().lower()
                if content_top_setting in {"", "auto"}:
                    content_top = self._detect_browser_content_top(source_rgb)
                else:
                    try:
                        content_top = max(0, int(content_top_setting))
                    except ValueError:
                        content_top = self._detect_browser_content_top(source_rgb)
                image = source_rgb.convert("L")
                bottom = max(content_top + 1, image.height - bottom_ignore)
                content = image.crop((0, min(content_top, image.height - 1), image.width, bottom))
                if sample_width > 0 and content.width > sample_width:
                    ratio = sample_width / content.width
                    content = content.resize(
                        (sample_width, max(1, int(content.height * ratio))),
                        Image.Resampling.BILINEAR,
                    )
                stats = ImageStat.Stat(content)
                edge_image = content.filter(ImageFilter.FIND_EDGES)
                if edge_image.width > 2 and edge_image.height > 2:
                    # FIND_EDGES가 단색 이미지 외곽에 만드는 인공 경계는 품질 계산에서 제외한다.
                    edge_image = edge_image.crop((1, 1, edge_image.width - 1, edge_image.height - 1))
                edge_mean = ImageStat.Stat(edge_image).mean[0]
                histogram = content.histogram()
                pixel_count = max(1, sum(histogram))
                dominant_ratio = max(histogram) / pixel_count
                stddev = stats.stddev[0]
        except Exception as exc:
            logger.debug("Screen quality check skipped", error=str(exc), image_path=str(image_path))
            return {"low_information": False, "reason": "quality_check_error"}

        settings = get_settings().vision
        max_stddev = settings.page_blank_max_stddev
        max_edge_mean = settings.page_blank_max_edge_mean
        min_dominant_ratio = settings.page_blank_min_dominant_ratio
        low_information = (
            stddev <= max_stddev
            and edge_mean <= max_edge_mean
            and dominant_ratio >= min_dominant_ratio
        )
        return {
            "low_information": low_information,
            "reason": "low_information_page" if low_information else "page_content_present",
            "stddev": round(stddev, 3),
            "edge_mean": round(edge_mean, 3),
            "dominant_ratio": round(dominant_ratio, 4),
        }

    def capture_usable_screen(
        self,
        max_attempts: Optional[int] = None,
        *,
        initial_wait_sec: Optional[float] = None,
    ) -> Path:
        """단색 빈 본문이면 한 경로에서 기다리고 마지막 화면만 보존한다."""
        settings = get_settings().vision
        retry_interval = settings.page_capture_retry_sec
        ready_timeout = settings.page_ready_timeout_sec
        if max_attempts is None:
            polling_interval = max(0.05, retry_interval)
            max_attempts = max(1, math.ceil(max(0.0, ready_timeout) / polling_interval) + 1)
            deadline = time.monotonic() + max(0.0, ready_timeout)
        else:
            max_attempts = max(1, int(max_attempts))
            deadline = None
        last_path: Path | None = None
        for attempt in range(1, max_attempts + 1):
            # 로딩 재관찰은 같은 파일을 덮어써 빈 화면 산출물이 쌓이지 않게 한다.
            filename = last_path.name if last_path is not None else None
            if initial_wait_sec is None:
                last_path = self.capture_screen(filename=filename)
            else:
                last_path = self.capture_screen(filename=filename, initial_wait_sec=initial_wait_sec)
            quality = self.screen_quality(last_path)
            wait_stable = getattr(self, "_wait_stable", None)
            stability = dict(
                getattr(wait_stable, "last_wait_result", {}) or {}
            )
            if stability:
                quality.update(
                    {
                        "stable": bool(stability.get("stable")),
                        "stability_reason": str(
                            stability.get("reason") or ""
                        ),
                        "stability_probe_count": int(
                            stability.get("probe_count") or 0
                        ),
                        "stability_confirmations": int(
                            stability.get("stable_frames") or 0
                        ),
                        "stability_diff_percent": stability.get(
                            "diff_percent"
                        ),
                    }
                )
            self.last_capture_quality = dict(quality)
            logger.info("Screen quality checked", attempt=attempt, max_attempts=max_attempts, **quality)
            if (
                not quality.get("low_information")
                and quality.get("stable", True)
            ):
                return last_path
            if deadline is not None and time.monotonic() >= deadline:
                break
            if attempt < max_attempts and retry_interval > 0:
                wait_sec = retry_interval
                if deadline is not None:
                    wait_sec = min(wait_sec, max(0.0, deadline - time.monotonic()))
                if wait_sec > 0:
                    time.sleep(wait_sec)
        return last_path

    def get_current_url(self) -> str:
        """활성 브라우저의 주소창 URL을 클립보드 경유로 읽습니다."""
        url = ""
        try:
            import platform
            import pyautogui
            import pyperclip

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
                self.release_address_bar_focus(pyautogui, key_pause=key_pause)
                pyautogui.PAUSE = old_pause
            if url.startswith(("http://", "https://")):
                self._last_url = url
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

    def release_address_bar_focus(self, pyautogui_module=None, key_pause: float = 0.02) -> None:
        """주소창 URL 복사 후 남는 브라우저 툴바 포커스를 페이지 쪽으로 되돌립니다."""
        try:
            if pyautogui_module is None:
                import pyautogui as pyautogui_module

            old_pause = pyautogui_module.PAUSE
            pyautogui_module.PAUSE = min(old_pause, key_pause)
            try:
                pyautogui_module.press("esc")
                pyautogui_module.press("esc")
            finally:
                pyautogui_module.PAUSE = old_pause
        except Exception as e:
            logger.debug("Failed to release browser address bar focus", error=str(e))

    def _image_signature(self, image_path: Path) -> str:
        try:
            with Image.open(image_path) as img:
                thumb = img.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
                return hashlib.sha1(thumb.tobytes()).hexdigest()
        except Exception as e:
            logger.debug("Failed to build image signature", error=str(e))
            return ""

    def _cache_analysis(self, key: str, analysis: Dict[str, Any]) -> None:
        if not key or self._analysis_cache_limit <= 0:
            return
        self._analysis_cache[key] = {
            "markers": [dict(marker) for marker in analysis.get("markers", [])],
            "marked_image": analysis.get("marked_image", ""),
            "content_top": int(analysis.get("content_top", 0) or 0),
        }
        if key in self._analysis_cache_order:
            self._analysis_cache_order.remove(key)
        self._analysis_cache_order.append(key)
        while len(self._analysis_cache_order) > self._analysis_cache_limit:
            old_key = self._analysis_cache_order.pop(0)
            self._analysis_cache.pop(old_key, None)

    def _prepare_som_image(self, image_path: Path) -> tuple[Path, int]:
        crop_setting = get_settings().vision.som_crop_top.strip().lower()
        try:
            with Image.open(image_path) as img:
                if crop_setting in {"", "auto"}:
                    crop_top = self._detect_browser_content_top(img)
                else:
                    try:
                        crop_top = max(0, int(crop_setting))
                    except ValueError:
                        crop_top = self._detect_browser_content_top(img)
                if crop_top <= 0:
                    return image_path, 0
                if img.height <= crop_top + 400:
                    return image_path, 0
                cropped_path = image_path.with_name(f"som_{image_path.name}")
                cropped = img.crop((0, crop_top, img.width, img.height))
                if cropped_path.suffix.lower() in (".jpg", ".jpeg"):
                    cropped.save(cropped_path, "JPEG", quality=80)
                else:
                    cropped.save(cropped_path)
                return cropped_path, crop_top
        except Exception as e:
            logger.debug("Failed to crop browser chrome for SoM", error=str(e))
            return image_path, 0

    def _detect_browser_content_top(self, image: Image.Image) -> int:
        """브라우저 chrome과 페이지 사이의 전체 폭 수평 경계를 찾는다."""

        settings = get_settings().vision
        fallback_top = settings.som_crop_fallback_top_px
        min_y = settings.som_crop_scan_min_y
        max_y = min(
            settings.som_crop_scan_max_y,
            max(0, image.height - 400),
        )
        if max_y <= min_y:
            return fallback_top

        sample_width = settings.som_crop_sample_width
        sample = image.convert("RGB")
        if sample.width > sample_width:
            sample = sample.resize((sample_width, sample.height), Image.Resampling.BILINEAR)

        row_means = [
            ImageStat.Stat(sample.crop((0, y, sample.width, y + 1))).mean
            for y in range(max_y + 1)
        ]
        best_y = fallback_top
        best_delta = 0.0
        for y in range(max(1, min_y), max_y + 1):
            delta = sum(abs(row_means[y][channel] - row_means[y - 1][channel]) for channel in range(3))
            if delta > best_delta:
                best_delta = delta
                best_y = y

        min_delta = settings.som_crop_min_row_delta
        if best_delta < min_delta:
            logger.debug(
                "Browser content boundary was weak; using crop fallback",
                best_delta=round(best_delta, 2),
                fallback_top=fallback_top,
            )
            return fallback_top
        logger.info(
            "Detected browser content boundary",
            crop_top=best_y,
            row_delta=round(best_delta, 2),
        )
        return best_y

    def analyze_ui(self, image_path: Path) -> Dict[str, Any]:
        """OmniParser와 PaddleOCR 결과를 물리 좌표가 있는 SoM 마커로 변환한다."""

        if not image_path.exists():
            logger.error("Image file not found for UI analysis", image_path=str(image_path))
            raise FileNotFoundError(f"Image not found: {image_path}")

        cache_key = self._image_signature(image_path)
        if cache_key and cache_key in self._analysis_cache:
            cached = self._analysis_cache[cache_key]
            logger.info("UI analysis cache hit", markers_count=len(cached.get("markers", [])))
            return {
                "markers": [dict(marker) for marker in cached.get("markers", [])],
                "marked_image": cached.get("marked_image", ""),
                "content_top": int(cached.get("content_top", 0) or 0),
                "analysis_mode": str(cached.get("analysis_mode") or "full"),
            }

        # 1. 로컬 SoM 엔진 실행 (마킹 이미지 합성 및 좌표 추출)
        som_image_path, crop_top = self._prepare_som_image(image_path)
        try:
            marked_filename = f"marked_{image_path.name}"
            marked_path, marker_bboxes, final_elements = self.som_engine.process_image(
                som_image_path,
                output_filename=marked_filename,
            )
            if crop_top:
                for bbox in marker_bboxes.values():
                    bbox[1] += crop_top
                    bbox[3] += crop_top
                logger.info("Applied browser chrome crop for SoM", crop_top=crop_top)
        except Exception as som_err:
            logger.error("Local SoM processing failed", error=str(som_err))
            return {"markers": [], "marked_image": ""}
        finally:
            if som_image_path != image_path:
                try:
                    som_image_path.unlink(missing_ok=True)
                except Exception:
                    pass

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
        analysis = {
            "markers": markers,
            "marked_image": str(marked_path),
            "content_top": crop_top,
            "analysis_mode": "full",
        }
        self._cache_analysis(cache_key, analysis)
        return analysis
