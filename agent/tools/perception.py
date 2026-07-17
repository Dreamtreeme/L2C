import datetime
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mss
import mss.tools
import pygetwindow as gw
from PIL import Image, ImageFilter, ImageStat

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
        self._analysis_cache_limit = int(os.getenv("VISION_UI_ANALYSIS_CACHE_LIMIT", "8"))

        # WaitStable은 PerceptionEngine을 역참조하므로 순환 import를 피하기 위해 lazy 로딩합니다.
        from agent.utils.wait_stable import WaitStable
        self._wait_stable = WaitStable(self)

        logger.info("PerceptionEngine initialized with SomEngine", screenshot_dir=str(self.screenshot_dir))

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return max(0, int(os.getenv(name, str(default))))
        except ValueError:
            return default

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
        return {
            "top": top,
            "left": win.left + border,
            "width": win.width - (border * 2),
            "height": win.height - border - (border if win.isMaximized else 0),
        }

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
        페이지의 실제 로딩 완료 여부는 이후 전환 계약 검사에서 판단합니다.

        Args:
            filename: 저장할 파일명. 입력하지 않으면 타임스탬프 기반 자동 생성.

        Returns:
            저장된 스크린샷 이미지의 절대 경로 (Path 객체)
        """
        # 액션 효과가 캡처에 반영되기 시작할 짧은 시간만 확보합니다.
        if initial_wait_sec is None:
            initial_wait_sec = self._env_float("VISION_CAPTURE_INITIAL_WAIT_SEC", 0.16)
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

    def screen_quality(self, image_path: Path) -> Dict[str, Any]:
        """브라우저 본문이 단색 빈 화면에 가까운지 저비용 이미지 지표로 검사한다."""
        bottom_ignore = self._env_int("VISION_PAGE_CONTENT_BOTTOM_IGNORE_PX", 80)
        sample_width = self._env_int("VISION_PAGE_QUALITY_SAMPLE_WIDTH", 240)
        try:
            with Image.open(image_path) as source:
                source_rgb = source.convert("RGB")
                content_top_setting = os.getenv("VISION_PAGE_CONTENT_TOP_PX", "auto").strip().lower()
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

        max_stddev = self._env_float("VISION_PAGE_BLANK_MAX_STDDEV", 6.0)
        max_edge_mean = self._env_float("VISION_PAGE_BLANK_MAX_EDGE_MEAN", 1.0)
        min_dominant_ratio = self._env_float("VISION_PAGE_BLANK_MIN_DOMINANT_RATIO", 0.98)
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

    def capture_usable_screen(self, max_attempts: Optional[int] = None) -> Path:
        """단색 빈 본문이면 OCR 전에 짧게 재캡처하고 마지막 화면을 반환한다."""
        if max_attempts is None:
            max_attempts = self._env_int("VISION_PAGE_CAPTURE_MAX_ATTEMPTS", 4)
        max_attempts = max(1, int(max_attempts))
        retry_interval = self._env_float("VISION_PAGE_CAPTURE_RETRY_SEC", 0.4)
        last_path: Path | None = None
        for attempt in range(1, max_attempts + 1):
            filename = None
            if attempt > 1:
                filename = f"screen_retry_{int(time.time() * 1000)}_{attempt}.png"
            last_path = self.capture_screen(filename=filename)
            quality = self.screen_quality(last_path)
            self.last_capture_quality = dict(quality)
            logger.info("Screen quality checked", attempt=attempt, max_attempts=max_attempts, **quality)
            if not quality.get("low_information"):
                return last_path
            if attempt < max_attempts and retry_interval > 0:
                time.sleep(retry_interval)
        return last_path

    def get_current_url(self) -> str:
        """활성 브라우저의 주소창 URL을 클립보드 경유로 읽습니다."""
        try:
            import platform
            import pyautogui
            import pyperclip

            old_pause = pyautogui.PAUSE
            modifier = "command" if platform.system() == "Darwin" else "ctrl"
            key_pause = self._env_float("VISION_URL_KEY_PAUSE_SEC", 0.015)
            copy_wait = self._env_float("VISION_URL_COPY_WAIT_SEC", 0.015)
            pyautogui.PAUSE = min(old_pause, key_pause)
            try:
                pyperclip.copy("")
                pyautogui.hotkey(modifier, "l")
                pyautogui.hotkey(modifier, "c")
                if copy_wait > 0:
                    time.sleep(copy_wait)
                url = (pyperclip.paste() or "").strip()
            finally:
                self.release_address_bar_focus(pyautogui, key_pause=key_pause)
                pyautogui.PAUSE = old_pause
            if url.startswith(("http://", "https://")):
                self._last_url = url
                return url
        except Exception as e:
            logger.debug("Failed to read current browser URL", error=str(e))
        return self._last_url

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
            "original_image": analysis.get("original_image", ""),
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
        crop_setting = os.getenv("VISION_SOM_CROP_TOP", "auto").strip().lower()
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

        fallback_top = self._env_int("VISION_SOM_CROP_FALLBACK_TOP_PX", 140)
        min_y = self._env_int("VISION_SOM_CROP_SCAN_MIN_Y", 80)
        max_y = min(
            self._env_int("VISION_SOM_CROP_SCAN_MAX_Y", 320),
            max(0, image.height - 400),
        )
        if max_y <= min_y:
            return fallback_top

        sample_width = max(32, self._env_int("VISION_SOM_CROP_SAMPLE_WIDTH", 256))
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

        min_delta = self._env_float("VISION_SOM_CROP_MIN_ROW_DELTA", 60.0)
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
        """
        Set-of-Marks (SoM) 기반의 UI 분석 엔진입니다.
        OmniParser YOLO 및 PaddleOCR로 마킹 이미지를 생성한 뒤 VLM(Gemini/Ollama)을 호출하여
        각 마커 ID의 서비스 상 용도를 캡셔닝하고 물리 좌표와 맵핑하여 반환합니다.
        
        Args:
            image_path: 원본 스크린샷 이미지 경로
            
        Returns:
            UI 마커의 ID, 텍스트, 바운딩 박스(bbox) 목록을 담은 딕셔너리
        """
        import json
        
        if not image_path.exists():
            logger.error("Image file not found for UI analysis", image_path=str(image_path))
            raise FileNotFoundError(f"Image not found: {image_path}")

        cache_key = self._image_signature(image_path)
        if cache_key and cache_key in self._analysis_cache:
            cached = self._analysis_cache[cache_key]
            logger.info("UI analysis cache hit", markers_count=len(cached.get("markers", [])))
            return {
                "markers": [dict(marker) for marker in cached.get("markers", [])],
                "original_image": str(image_path),
                "marked_image": cached.get("marked_image", ""),
                "content_top": int(cached.get("content_top", 0) or 0),
                "analysis_mode": str(cached.get("analysis_mode") or "full"),
            }

        # 1. 로컬 SoM 엔진 실행 (마킹 이미지 합성 및 좌표 추출)
        som_image_path, crop_top = self._prepare_som_image(image_path)
        try:
            marked_filename = f"marked_{image_path.name}"
            marked_path, marker_coords, marker_bboxes, final_elements = self.som_engine.process_image(
                som_image_path,
                output_filename=marked_filename,
            )
            if crop_top:
                for bbox in marker_bboxes.values():
                    bbox[1] += crop_top
                    bbox[3] += crop_top
                for coords in marker_coords.values():
                    coords[1] += crop_top
                logger.info("Applied browser chrome crop for SoM", crop_top=crop_top)
        except Exception as som_err:
            logger.error("Local SoM processing failed", error=str(som_err))
            return {"markers": [], "original_image": str(image_path)}
        finally:
            if som_image_path != image_path:
                try:
                    som_image_path.unlink(missing_ok=True)
                except Exception:
                    pass

        skip_vlm_caption = os.getenv("SKIP_VLM_CAPTION", "true").lower() == "true"
        elements = []

        if skip_vlm_caption:
            logger.info("Bypassing VLM captioning node as SKIP_VLM_CAPTION is set to true.")
        else:
            # 2. 마킹된 이미지 로드 및 리사이징 (JPEG 압축 및 VLM 최적화)
            try:
                from agent.utils.image_utils import image_to_base64_jpeg
                # fast=False: LANCZOS 리사이징 + quality=80 으로 화질 우선 (캡셔닝 정확도)
                base64_image = image_to_base64_jpeg(marked_path, max_dim=1024, quality=80, fast=False)
            except Exception as img_err:
                logger.error("Failed to load and resize marked image", error=str(img_err))
                return {"markers": [], "original_image": str(image_path)}

            # 3. VLM 프롬프트 작성 (ID 매핑 요청 - 토큰 길이 및 속도 최적화 버전)
            prompt = """
Analyze this UI screenshot of a Korean website, which has numbered markers on it (like [0], [1], [2], ...).
Describe ONLY the most important clickable/interactable elements (e.g. GNB menu items, major buttons, input fields, search results, tabs).

Optimization rules:
1. Focus ONLY on interactive/clickable elements. Ignore background static texts, tiny decorations, or unidentifiable symbols.
2. Keep descriptions extremely short and concise (e.g., 2-4 words maximum, like "검색창", "구글 로그인", "데이터 분석가 채용").
3. Limit the response to at most 35-40 of the most significant elements to keep it compact.

You MUST return a single JSON object with the key "elements".
Each object in the "elements" array must have:
- "id": integer corresponding to the marker number in the image
- "text": short description of the element (e.g. "구글 로그인")

Example output format:
{
  "elements": [
    {"id": 0, "text": "검색창"},
    {"id": 1, "text": "회원가입"}
  ]
}
"""

            api_key = os.getenv("GEMINI_API_KEY")

            # 4. Gemini Flash 호출 시도 (langchain_google_genai — llm_engine.py와 동일한 클라이언트)
            if api_key:
                try:
                    from agent.application.model_clients import get_google_chat_model
                    from langchain_core.messages import HumanMessage
                    logger.info("Captioning UI elements via Gemini Flash SoM...")
                    llm = get_google_chat_model("gemini-3.5-flash", temperature=0.1)
                    message = HumanMessage(content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ])
                    from agent.application.run_context import invoke_with_metrics

                    response = invoke_with_metrics(
                        llm,
                        [message],
                        "vision_caption",
                    )
                    output = response.content
                    if isinstance(output, list):
                        output = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in output)
                    # 마크다운 펜스가 있으면 제거
                    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", output, re.DOTALL)
                    json_str = m.group(1).strip() if m else output.strip()
                    data = json.loads(json_str)
                    elements = data.get("elements", [])
                    logger.info("Gemini SoM captioning completed successfully", elements_count=len(elements))
                except Exception as gemini_err:
                    logger.warning("Gemini SoM captioning failed, falling back to local Ollama", error=str(gemini_err))

            # 5. 로컬 Ollama (Qwen2.5-VL) Fallback 호출 시도 (ollama 클라이언트 — llm_engine.py와 동일)
            if not elements:
                logger.info("Captioning UI elements via local Ollama SoM (Fallback)...")
                try:
                    import ollama as _ollama
                    from shared.config import OLLAMA_HOST
                    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
                    client = _ollama.Client(host=OLLAMA_HOST)
                    from agent.application.run_context import observe_external_llm_call

                    with observe_external_llm_call(
                        component="vision_caption",
                        provider="ollama",
                        model=model_name,
                    ) as observation:
                        resp = client.generate(
                            model=model_name,
                            prompt=prompt,
                            images=[base64_image],
                            stream=False,
                            options={"num_ctx": 4096, "num_predict": 1024, "temperature": 0.1}
                        )
                        usage_source = (
                            resp
                            if isinstance(resp, dict)
                            else {
                                "prompt_eval_count": getattr(resp, "prompt_eval_count", 0),
                                "eval_count": getattr(resp, "eval_count", 0),
                            }
                        )
                        observation.set_usage(
                            {
                                "input_tokens": usage_source.get("prompt_eval_count", 0),
                                "output_tokens": usage_source.get("eval_count", 0),
                            }
                        )
                    # ollama 버전에 따라 dict 또는 GenerateResponse 객체로 반환됨
                    result_text = (resp.get("response", "") if isinstance(resp, dict) else getattr(resp, "response", "")) or ""
                    thinking_text = (resp.get("thinking", "") if isinstance(resp, dict) else getattr(resp, "thinking", "")) or ""
                    parse_target = result_text.strip() or thinking_text.strip()

                    if parse_target:
                        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", parse_target, re.DOTALL)
                        json_str = m.group(1).strip() if m else parse_target.strip()
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict) and "elements" in parsed:
                                elements = parsed["elements"]
                            elif isinstance(parsed, list):
                                elements = parsed
                        except Exception:
                            pass
                    logger.info("Ollama SoM captioning completed", elements_count=len(elements))
                except Exception as ollama_err:
                    logger.error("Failed to caption UI elements via Ollama", error=str(ollama_err))

        # 6. 매핑 정보 병합 및 안전한 Fallback 매핑 (VLM 누락 마커 처리)
        id_to_text = {}
        if skip_vlm_caption:
            for marker_id, elem in enumerate(final_elements):
                local_text = elem.get("text", "")
                elem_type = elem.get("type", "element")
                if elem_type == "text" and local_text:
                    id_to_text[marker_id] = local_text
                else:
                    id_to_text[marker_id] = f"상호작용 가능한 요소 ({elem_type})"
        else:
            if elements:
                for elem in elements:
                    if isinstance(elem, dict) and "id" in elem:
                        try:
                            id_to_text[int(elem["id"])] = elem.get("text", "상호작용 가능한 요소")
                        except ValueError:
                            continue

        markers = []
        for marker_id, bbox in marker_bboxes.items():
            text = id_to_text.get(marker_id, "상호작용 가능한 요소 (미식별)")
            elem = final_elements[marker_id] if marker_id < len(final_elements) else {}
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
            "original_image": str(image_path),
            "marked_image": str(marked_path),
            "content_top": crop_top,
            "analysis_mode": "full",
        }
        self._cache_analysis(cache_key, analysis)
        return analysis
