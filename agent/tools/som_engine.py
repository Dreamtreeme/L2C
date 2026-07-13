import os
import json
import queue
import site
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from agent.utils.logger import logger


_DLL_DIRECTORY_HANDLES = []


class SomEngine:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self._configure_runtime_paths()

        self.paddleocr_dir = self._resolve_paddleocr_base_dir()
        self.paddleocr_dir.mkdir(parents=True, exist_ok=True)
        os.environ["PADDLE_OCR_BASE_DIR"] = str(self.paddleocr_dir)
        self._ocr_worker = None
        self._ocr_worker_stdout_queue = None
        self._ocr_worker_stderr_lines = deque(maxlen=40)
        self._ocr_worker_last_phase: Dict[str, Any] = {}
        self._ocr_worker_last_result_timings: Dict[str, Any] = {}
        self._ocr_worker_generation = 0
        self._ocr_worker_request_count = 0
        self._ocr_worker_lifecycle_lock = threading.RLock()

        self.model_dir = self.root_dir / "models" / "omniparser"
        self.model_path = self.model_dir / "icon_detect" / "model.pt"

        self._ensure_model_downloaded()

        from ultralytics import YOLO

        logger.info("Loading local YOLOv8 OmniParser model", model_path=str(self.model_path))
        self.yolo_model = YOLO(str(self.model_path))

        logger.info("SomEngine will invoke PaddleOCR in an isolated worker", cache_dir=str(self.paddleocr_dir))
        logger.info("SomEngine initialization complete")

    def __del__(self):
        self._stop_ocr_worker()

    def _configure_runtime_paths(self) -> None:
        yolo_config_dir = Path(os.getenv("YOLO_CONFIG_DIR", self.root_dir / ".cache" / "ultralytics"))
        yolo_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)

        for path in self._candidate_cuda_dll_dirs():
            try:
                exists = path.exists()
            except OSError:
                exists = False
            if not exists:
                continue
            self._prepend_process_path(path)
            if hasattr(os, "add_dll_directory"):
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))
            logger.debug("Added CUDA DLL search path", path=str(path))

    def _candidate_cuda_dll_dirs(self) -> List[Path]:
        paths: List[Path] = []
        for key in ("PADDLE_CUDA_BIN_DIR", "PADDLE_CUDNN_BIN_DIR", "CUDA_PATH", "CUDA_PATH_V11_8"):
            value = os.getenv(key)
            if value:
                base = Path(value)
                paths.append(base / "bin" if base.name.lower() != "bin" else base)

        for site_dir in site.getsitepackages():
            nvidia_dir = Path(site_dir) / "nvidia"
            paths.extend(
                [
                    nvidia_dir / "cudnn" / "bin",
                    nvidia_dir / "cublas" / "bin",
                    nvidia_dir / "cuda_runtime" / "bin",
                    nvidia_dir / "cuda_nvrtc" / "bin",
                ]
            )
        user_site = site.getusersitepackages()
        if user_site:
            nvidia_dir = Path(user_site) / "nvidia"
            paths.extend(
                [
                    nvidia_dir / "cudnn" / "bin",
                    nvidia_dir / "cublas" / "bin",
                    nvidia_dir / "cuda_runtime" / "bin",
                    nvidia_dir / "cuda_nvrtc" / "bin",
                ]
            )

        unique_paths: List[Path] = []
        seen = set()
        for path in paths:
            key = str(path).lower()
            if key not in seen:
                seen.add(key)
                unique_paths.append(path)
        return unique_paths

    @staticmethod
    def _prepend_process_path(path: Path) -> None:
        path_text = str(path)
        current = os.environ.get("PATH", "")
        parts = [part for part in current.split(os.pathsep) if part]
        if path_text.lower() not in {part.lower() for part in parts}:
            os.environ["PATH"] = path_text + os.pathsep + current

    def _resolve_paddleocr_base_dir(self) -> Path:
        configured = os.getenv("PADDLE_OCR_BASE_DIR")
        if configured:
            return Path(configured)

        project_cache = self.root_dir / ".cache" / "paddleocr"
        if (project_cache / "whl").exists():
            return project_cache

        home_cache = Path.home() / ".paddleocr"
        if (home_cache / "whl").exists():
            return home_cache

        return project_cache

    def _paddleocr_model_dirs(self) -> Dict[str, str]:
        whl_dir = self.paddleocr_dir / "whl"
        dirs = {
            "det_model_dir": whl_dir / "det" / "ml" / "Multilingual_PP-OCRv3_det_infer",
            "rec_model_dir": whl_dir / "rec" / "korean" / "korean_PP-OCRv4_rec_infer",
            "cls_model_dir": whl_dir / "cls" / "ch_ppocr_mobile_v2.0_cls_infer",
        }
        if all(path.exists() for path in dirs.values()):
            return {key: str(path) for key, path in dirs.items()}
        return {}

    @staticmethod
    def _should_use_paddle_gpu(torch_cuda_available: bool) -> bool:
        raw = os.getenv("PADDLEOCR_USE_GPU")
        if raw is not None:
            return raw.strip().lower() not in {"0", "false", "no", "off"}
        return torch_cuda_available

    def _ocr_worker_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env["PADDLE_OCR_BASE_DIR"] = str(self.paddleocr_dir)
        return env

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return max(0.0, float(os.getenv(name, str(default))))
        except ValueError:
            return default

    def _ocr_worker_start_timeout_sec(self) -> float:
        return max(1.0, self._env_float("SOM_OCR_WORKER_START_TIMEOUT_SEC", 45.0))

    def _ocr_worker_request_timeout_sec(self) -> float:
        return max(1.0, self._env_float("SOM_OCR_REQUEST_TIMEOUT_SEC", 20.0))

    def _ocr_worker_attempts(self) -> int:
        return max(1, self._env_int("SOM_OCR_WORKER_MAX_ATTEMPTS", 2))

    def _ocr_lifecycle_lock(self):
        """테스트용 비정상 생성 경로에서도 OCR 작업자 잠금을 지연 생성한다."""

        lock = getattr(self, "_ocr_worker_lifecycle_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._ocr_worker_lifecycle_lock = lock
        return lock

    def ensure_ocr_worker_ready(self):
        """재사용 OCR 작업자가 요청을 받을 수 있을 때까지 대기한다."""

        use_worker = os.getenv("SOM_OCR_WORKER_REUSE", "true").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if not use_worker:
            raise RuntimeError("OCR worker readiness requires SOM_OCR_WORKER_REUSE")
        return self._start_ocr_worker()

    def _read_ocr_worker_stdout(self, worker: subprocess.Popen, generation: int, output_queue: queue.Queue) -> None:
        try:
            assert worker.stdout is not None
            while True:
                line = worker.stdout.readline()
                output_queue.put((generation, line))
                if line == "":
                    return
        except Exception:
            output_queue.put((generation, ""))

    def _read_ocr_worker_stderr(self, worker: subprocess.Popen, generation: int) -> None:
        try:
            assert worker.stderr is not None
            for line in worker.stderr:
                if generation != self._ocr_worker_generation:
                    return
                stripped = line.strip()
                if stripped:
                    self._ocr_worker_stderr_lines.append(stripped[:500])
        except Exception:
            return

    def _ocr_worker_diagnostics(self, request_id: str = "") -> dict[str, Any]:
        phase = dict(getattr(self, "_ocr_worker_last_phase", {}) or {})
        if request_id and phase.get("request_id") not in {"", request_id}:
            phase = {}
        stderr_lines = list(getattr(self, "_ocr_worker_stderr_lines", []) or [])
        return {
            "last_phase": str(phase.get("phase") or ""),
            "last_phase_details": phase,
            "worker_stderr": stderr_lines[-8:],
        }

    def _next_ocr_worker_line(self, timeout_sec: float) -> str | None:
        output_queue = self._ocr_worker_stdout_queue
        if output_queue is None:
            return None
        deadline = time.monotonic() + timeout_sec
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                generation, line = output_queue.get(timeout=remaining)
            except queue.Empty:
                return None
            if generation == self._ocr_worker_generation:
                return line

    def _start_ocr_worker(self):
        with self._ocr_lifecycle_lock():
            return self._start_ocr_worker_locked()

    def _start_ocr_worker_locked(self):
        if self._ocr_worker and self._ocr_worker.poll() is None:
            return self._ocr_worker

        runner_script = Path(__file__).parent / "paddle_ocr_runner.py"
        logger.info("Starting isolated PaddleOCR worker", script=str(runner_script))
        self._ocr_worker_generation += 1
        generation = self._ocr_worker_generation
        self._ocr_worker_stdout_queue = queue.Queue()
        self._ocr_worker_stderr_lines = deque(maxlen=40)
        self._ocr_worker_last_phase = {}
        self._ocr_worker_last_result_timings = {}
        self._ocr_worker = subprocess.Popen(
            [sys.executable, str(runner_script), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
            env=self._ocr_worker_env(),
        )

        assert self._ocr_worker.stdout is not None
        threading.Thread(
            target=self._read_ocr_worker_stdout,
            args=(self._ocr_worker, generation, self._ocr_worker_stdout_queue),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_ocr_worker_stderr,
            args=(self._ocr_worker, generation),
            daemon=True,
        ).start()
        start_timeout = self._ocr_worker_start_timeout_sec()
        started = time.monotonic()
        while True:
            remaining = start_timeout - (time.monotonic() - started)
            if remaining <= 0:
                self._stop_ocr_worker()
                raise TimeoutError(f"PaddleOCR worker startup timed out after {start_timeout:.1f}s")
            line = self._next_ocr_worker_line(remaining)
            if line is None:
                self._stop_ocr_worker()
                raise TimeoutError(f"PaddleOCR worker startup timed out after {start_timeout:.1f}s")
            if line == "":
                self._ocr_worker = None
                raise RuntimeError("PaddleOCR worker exited during startup")
            stripped = line.strip()
            if stripped == "__OCR_WORKER_READY__":
                self._ocr_worker_request_count = 0
                startup_duration = time.monotonic() - started
                logger.info(
                    "PaddleOCR worker ready",
                    pid=self._ocr_worker.pid,
                    startup=f"{startup_duration:.2f}s",
                )
                from agent.application.run_context import current_run_context

                context = current_run_context()
                if context is not None:
                    context.record_step(
                        "ocr_startup",
                        startup_duration,
                        pid=self._ocr_worker.pid,
                    )
                return self._ocr_worker
            if stripped.startswith("__OCR_WORKER_FAILED__"):
                diagnostics = self._ocr_worker_diagnostics()
                self._ocr_worker = None
                raise RuntimeError(
                    "PaddleOCR worker failed during startup"
                    + (f": {diagnostics['worker_stderr'][-1]}" if diagnostics["worker_stderr"] else "")
                )
            if stripped:
                logger.debug("PaddleOCR worker startup output", line=stripped[:200])

    def _stop_ocr_worker(self) -> None:
        with self._ocr_lifecycle_lock():
            self._stop_ocr_worker_locked()

    def _stop_ocr_worker_locked(self) -> None:
        worker = getattr(self, "_ocr_worker", None)
        self._ocr_worker = None
        self._ocr_worker_stdout_queue = None
        self._ocr_worker_last_phase = {}
        self._ocr_worker_request_count = 0
        if not worker:
            return
        try:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=2)
        except Exception:
            pass

    def _run_paddle_ocr_worker(self, image_path: Path) -> List[Dict[str, Any]]:
        attempts = self._ocr_worker_attempts()
        request_timeout = self._ocr_worker_request_timeout_sec()
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            worker = self._start_ocr_worker()
            if not worker.stdin or not worker.stdout:
                raise RuntimeError("PaddleOCR worker pipes are unavailable")

            request_id = f"{self._ocr_worker_generation}:{time.time_ns()}"
            request = json.dumps(
                {"request_id": request_id, "image_path": str(image_path)},
                ensure_ascii=False,
            )
            started = time.monotonic()
            worker_timings: dict[str, Any] = {}
            self._ocr_worker_last_result_timings = {}
            try:
                worker.stdin.write(request + "\n")
                worker.stdin.flush()

                while True:
                    remaining = request_timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        diagnostics = self._ocr_worker_diagnostics(request_id)
                        raise TimeoutError(
                            f"PaddleOCR worker request timed out after {request_timeout:.1f}s "
                            f"(last_phase={diagnostics['last_phase'] or 'unknown'}, "
                            f"stderr={diagnostics['worker_stderr']})"
                        )
                    line = self._next_ocr_worker_line(remaining)
                    if line is None:
                        diagnostics = self._ocr_worker_diagnostics(request_id)
                        raise TimeoutError(
                            f"PaddleOCR worker request timed out after {request_timeout:.1f}s "
                            f"(last_phase={diagnostics['last_phase'] or 'unknown'}, "
                            f"stderr={diagnostics['worker_stderr']})"
                        )
                    if line == "":
                        raise RuntimeError("PaddleOCR worker exited before returning a result")
                    stripped = line.strip()
                    if stripped.startswith("__OCR_EVENT__"):
                        event_payload = json.loads(
                            stripped.removeprefix("__OCR_EVENT__").strip() or "{}"
                        )
                        if str(event_payload.get("request_id") or "") == request_id:
                            self._ocr_worker_last_phase = event_payload
                        continue
                    if stripped.startswith("__OCR_WORKER_ERROR__"):
                        error_payload = json.loads(
                            stripped.removeprefix("__OCR_WORKER_ERROR__").strip() or "{}"
                        )
                        if str(error_payload.get("request_id") or "") != request_id:
                            continue
                        raise RuntimeError(
                            "PaddleOCR worker error "
                            f"during {error_payload.get('phase') or 'unknown'}: "
                            f"{error_payload.get('error_type') or 'Error'}: "
                            f"{error_payload.get('error') or ''}"
                        )
                    if not stripped.startswith("__OCR_JSON_RESULT__"):
                        continue
                    payload = stripped.removeprefix("__OCR_JSON_RESULT__").strip()
                    decoded = json.loads(payload) if payload else {}
                    if isinstance(decoded, dict):
                        if str(decoded.get("request_id") or "") != request_id:
                            continue
                        results = list(decoded.get("results") or [])
                        worker_timings = dict(decoded.get("timings") or {})
                    else:
                        results = list(decoded or [])
                    duration = time.monotonic() - started
                    self._ocr_worker_last_result_timings = worker_timings
                    self._ocr_worker_request_count += 1
                    logger.info(
                        "PaddleOCR worker request completed",
                        pid=worker.pid,
                        attempt=attempt,
                        request_count=self._ocr_worker_request_count,
                        duration=f"{duration:.2f}s",
                        boxes=len(results),
                        worker_timings=worker_timings,
                    )
                    from agent.application.run_context import current_run_context

                    context = current_run_context()
                    if context is not None:
                        context.record_step(
                            "ocr_request",
                            duration,
                            pid=worker.pid,
                            attempt=attempt,
                            request_count=self._ocr_worker_request_count,
                            boxes=len(results),
                            success=True,
                            **worker_timings,
                        )
                    return results
            except Exception as exc:
                last_error = exc
                failed_duration = time.monotonic() - started
                logger.warning(
                    "PaddleOCR worker request failed",
                    attempt=attempt,
                    attempts=attempts,
                    error=str(exc),
                )
                from agent.application.run_context import current_run_context

                context = current_run_context()
                if context is not None:
                    context.record_step(
                        "ocr_request",
                        failed_duration,
                        pid=worker.pid,
                        attempt=attempt,
                        success=False,
                        error=str(exc)[:300],
                    )
                self._stop_ocr_worker()
                if attempt >= attempts:
                    raise

        if last_error:
            raise last_error
        return []

    def _run_paddle_ocr_once(self, image_path: Path) -> List[Dict[str, Any]]:
        runner_script = Path(__file__).parent / "paddle_ocr_runner.py"
        timeout = max(1.0, self._env_float("SOM_OCR_ONESHOT_TIMEOUT_SEC", 60.0))
        try:
            result = subprocess.run(
                [sys.executable, str(runner_script), str(image_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                env=self._ocr_worker_env(),
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("PaddleOCR one-shot runner timed out", timeout=f"{timeout:.1f}s")
            return []
        marker = "__OCR_JSON_START__"
        if marker not in result.stdout:
            if result.stderr:
                logger.warning("PaddleOCR one-shot runner produced stderr", stderr=result.stderr.strip())
            return []
        payload = result.stdout.split(marker, 1)[-1].strip()
        return json.loads(payload) if payload else []

    def _ensure_model_downloaded(self) -> None:
        if self.model_path.exists():
            return

        logger.info("YOLOv8 weights not found locally. Triggering Hugging Face download")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download

        try:
            downloaded_path = hf_hub_download(
                repo_id="microsoft/OmniParser-v2.0",
                filename="icon_detect/model.pt",
                local_dir=str(self.model_dir),
            )
            logger.info("Model weights downloaded successfully", downloaded_path=downloaded_path)
        except Exception as exc:
            logger.error("Failed to download model weights from Hugging Face", error=str(exc))
            raise

    def _get_area(self, box: List[float]) -> float:
        return (box[2] - box[0]) * (box[3] - box[1])

    def _get_intersection_area(self, box1: List[float], box2: List[float]) -> float:
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        return (x_right - x_left) * (y_bottom - y_top)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return max(0, int(os.getenv(name, str(default))))
        except ValueError:
            return default

    def _ocr_scale_for_image(self, width: int, height: int) -> float:
        if os.getenv("SOM_OCR_RESIZE", "true").strip().lower() in {"0", "false", "no", "off"}:
            return 1.0
        max_dim = self._env_int("SOM_OCR_MAX_DIM", 1152)
        if max_dim <= 0:
            return 1.0
        longest = max(width, height)
        if longest <= max_dim:
            return 1.0
        return max_dim / longest

    def _normalize_paddleocr_results(self, ocr_results: List, scale: float = 1.0) -> List[Dict]:
        raw_boxes = []
        if ocr_results and isinstance(ocr_results[0], dict):
            for item in ocr_results:
                confidence = float(item.get("confidence", item.get("conf", 0.0)))
                if confidence < 0.2:
                    continue
                bbox = item["bbox"]
                raw_boxes.append(
                    {
                        "bbox": [float(coord) / scale for coord in bbox],
                        "type": "text",
                        "text": str(item.get("text", "")),
                        "conf": confidence,
                    }
                )
            logger.debug("PaddleOCR text detection complete", count=len(raw_boxes))
            return raw_boxes

        for line in self._iter_paddleocr_lines(ocr_results):
            bbox = line[0]
            text = line[1][0]
            confidence = float(line[1][1])
            if confidence < 0.2:
                continue

            xs = [point[0] / scale for point in bbox]
            ys = [point[1] / scale for point in bbox]
            raw_boxes.append(
                {
                    "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                    "type": "text",
                    "text": text,
                    "conf": float(confidence),
                }
            )

        logger.debug("PaddleOCR text detection complete", count=len(raw_boxes))
        return raw_boxes

    @staticmethod
    def _iter_paddleocr_lines(ocr_results: List) -> List:
        if not ocr_results:
            return []
        first = ocr_results[0]
        if SomEngine._looks_like_paddleocr_line(first):
            return ocr_results
        lines = []
        for page in ocr_results:
            if not page:
                continue
            if SomEngine._looks_like_paddleocr_line(page):
                lines.append(page)
            else:
                lines.extend(line for line in page if SomEngine._looks_like_paddleocr_line(line))
        return lines

    @staticmethod
    def _looks_like_paddleocr_line(value: Any) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[0], (list, tuple))
            and isinstance(value[1], (list, tuple))
            and len(value[1]) >= 2
            and isinstance(value[1][0], str)
        )

    def _run_paddle_ocr(self, image: Image.Image | Path, scale: float = 1.0) -> List[Dict]:
        temp_path: Path | None = None
        if isinstance(image, Image.Image):
            fd, raw_temp = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            temp_path = Path(raw_temp)
            image.convert("RGB").save(temp_path, "JPEG", quality=90)
            image_path = temp_path
        else:
            image_path = image

        try:
            use_worker = os.getenv("SOM_OCR_WORKER_REUSE", "true").strip().lower() not in {"0", "false", "no", "off"}
            if use_worker:
                try:
                    results = self._run_paddle_ocr_worker(image_path)
                except Exception as exc:
                    logger.warning("PaddleOCR worker failed; falling back to one-shot runner", error=str(exc))
                    self._stop_ocr_worker()
                    results = self._run_paddle_ocr_once(image_path)
            else:
                results = self._run_paddle_ocr_once(image_path)
            return self._normalize_paddleocr_results(results, scale=scale)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _run_yolo(self, inference_img: Image.Image, scale: float) -> List[Dict]:
        raw_boxes = []
        try:
            yolo_results = self.yolo_model(inference_img, conf=0.15, verbose=False)
            if yolo_results and len(yolo_results) > 0:
                for box in yolo_results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    confidence = float(box.conf.item())
                    raw_boxes.append(
                        {
                            "bbox": [coord / scale for coord in coords],
                            "type": "icon",
                            "text": "icon",
                            "conf": confidence,
                        }
                    )
            logger.debug("YOLOv8 element detection complete", count=len(raw_boxes))
        except Exception as exc:
            logger.error("YOLOv8 inference failed", error=str(exc))
        return raw_boxes

    def _filter_overlaps(self, raw_boxes: List[Dict]) -> List[Dict]:
        sorted_boxes = sorted(raw_boxes, key=lambda item: self._get_area(item["bbox"]), reverse=True)
        final_elements = []

        for box in sorted_boxes:
            bbox = box["bbox"]
            area = self._get_area(bbox)
            if area <= 0:
                continue

            is_duplicate = False
            for kept in final_elements:
                intersection = self._get_intersection_area(bbox, kept["bbox"])
                if intersection <= 0:
                    continue
                smaller_area = min(area, self._get_area(kept["bbox"]))
                if intersection / smaller_area > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_elements.append(box)

        final_elements.sort(key=lambda item: (item["bbox"][1] // 20, item["bbox"][0]))
        logger.info("Overlap filtering complete", before=len(raw_boxes), after=len(final_elements))
        return final_elements

    def _draw_markers(
        self,
        img: Image.Image,
        final_elements: List[Dict],
    ) -> Tuple[Image.Image, Dict[int, List[int]], Dict[int, List[int]]]:
        marked_img = img.copy()
        draw = ImageDraw.Draw(marked_img)

        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()

        marker_coords: Dict[int, List[int]] = {}
        marker_bboxes: Dict[int, List[int]] = {}

        for marker_id, elem in enumerate(final_elements):
            bbox = elem["bbox"]
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            marker_coords[marker_id] = [(xmin + xmax) // 2, (ymin + ymax) // 2]
            marker_bboxes[marker_id] = [xmin, ymin, xmax, ymax]

            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 127, 80), width=2)

            label_text = f"[{marker_id}]"
            left, top, right, bottom = font.getbbox(label_text)
            text_w = right - left
            text_h = bottom - top

            tag_xmin = xmin
            tag_ymin = max(0, ymin - text_h - 4)
            tag_xmax = xmin + text_w + 6
            tag_ymax = tag_ymin + text_h + 4

            draw.rectangle([tag_xmin, tag_ymin, tag_xmax, tag_ymax], fill=(0, 0, 0))
            draw.text((tag_xmin + 3, tag_ymin + 1), label_text, fill=(255, 255, 255), font=font)

        return marked_img, marker_coords, marker_bboxes

    def process_image(
        self,
        image_path: Path,
        output_filename: str = "marked_screen.png",
    ) -> Tuple[Path, Dict[int, List[int]], Dict[int, List[int]], List[Dict[str, Any]]]:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        try:
            img = Image.open(image_path)
        except Exception as exc:
            logger.error("Failed to load image for processing", error=str(exc))
            raise

        original_w, original_h = img.size

        try:
            yolo_max_dim = int(os.getenv("SOM_INFERENCE_MAX_DIM", "1024"))
        except ValueError:
            yolo_max_dim = 1024

        if original_w > yolo_max_dim or original_h > yolo_max_dim:
            yolo_scale = yolo_max_dim / max(original_w, original_h)
            inference_img = img.resize(
                (int(original_w * yolo_scale), int(original_h * yolo_scale)),
                Image.Resampling.BILINEAR,
            )
            logger.debug(
                "Resized image for YOLO inference",
                scale=yolo_scale,
                original=(original_w, original_h),
                target=inference_img.size,
            )
        else:
            yolo_scale = 1.0
            inference_img = img

        ocr_scale = self._ocr_scale_for_image(original_w, original_h)
        ocr_image = img
        if ocr_scale != 1.0:
            ocr_image = img.resize(
                (int(original_w * ocr_scale), int(original_h * ocr_scale)),
                Image.Resampling.BILINEAR,
            )
            logger.info(
                "Resized image for OCR inference",
                scale=round(ocr_scale, 3),
                original=(original_w, original_h),
                target=ocr_image.size,
            )

        ocr_started = time.perf_counter()
        text_boxes = self._filter_overlaps(self._run_paddle_ocr(ocr_image, scale=ocr_scale))
        ocr_duration = time.perf_counter() - ocr_started
        yolo_started = time.perf_counter()
        icon_boxes = self._filter_overlaps(self._run_yolo(inference_img, yolo_scale))
        yolo_duration = time.perf_counter() - yolo_started
        final_elements = text_boxes + icon_boxes
        final_elements.sort(key=lambda item: (item["bbox"][1] // 20, item["bbox"][0]))
        logger.info(
            "Raw OCR/icon detections merged",
            text=len(text_boxes),
            icon=len(icon_boxes),
            total=len(final_elements),
        )
        marked_img, marker_coords, marker_bboxes = self._draw_markers(img, final_elements)

        output_path = image_path.parent / output_filename
        if marked_img.mode != "RGB":
            marked_img = marked_img.convert("RGB")
        if output_path.suffix.lower() in (".jpg", ".jpeg"):
            marked_img.save(output_path, "JPEG", quality=85)
        else:
            marked_img.save(output_path)

        logger.info(
            "Set-of-Marks image synthesized and saved successfully",
            output_path=str(output_path),
            markers_count=len(marker_coords),
        )
        logger.info(
            "SoM analysis stages completed",
            mode="full",
            ocr_duration_sec=round(ocr_duration, 6),
            yolo_duration_sec=round(yolo_duration, 6),
            text_markers=len(text_boxes),
            icon_markers=len(icon_boxes),
        )
        return output_path, marker_coords, marker_bboxes, final_elements
