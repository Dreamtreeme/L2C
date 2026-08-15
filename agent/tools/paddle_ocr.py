"""재사용 PaddleOCR 작업자와 문자 검출 경계."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image

from agent.config import get_settings
from agent.observability.run_context import current_run_context, observe_step
from agent.utils.logger import logger


class PaddleOcr:
    """PaddleOCR 하위 프로세스를 작업 동안 재사용해 문자를 검출한다."""

    def __init__(self, root_dir: Path | None = None):
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent.parent
        self.cache_dir = self._resolve_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._worker: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue | None = None
        self._stderr_lines: deque[str] = deque(maxlen=40)
        self._last_phase: dict[str, Any] = {}
        self._generation = 0
        self._request_count = 0
        self._lifecycle_lock = threading.RLock()

    def close(self) -> None:
        self._stop_worker()

    @property
    def worker_pid(self) -> int | None:
        worker = self._worker
        if worker is None or worker.poll() is not None:
            return None
        return int(worker.pid)

    def ensure_ready(self):
        return self._start_worker()

    def _resolve_cache_dir(self) -> Path:
        configured = get_settings().ocr.paddlex_cache_dir
        if configured:
            return configured
        project_cache = self.root_dir / ".cache" / "paddlex"
        if project_cache.exists():
            return project_cache
        home_cache = Path.home() / ".paddlex"
        return home_cache if home_cache.exists() else project_cache

    def _resolve_python(self) -> Path:
        python_path = get_settings().ocr.python_executable
        if not python_path.is_file():
            raise FileNotFoundError(
                "PaddleOCR 작업자 Python을 찾을 수 없습니다: "
                f"{python_path}. scripts/setup_runtime.ps1을 실행하거나 "
                "PADDLE_OCR_PYTHON을 설정하세요."
            )
        return python_path

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        settings = get_settings().ocr
        env["PADDLE_PDX_CACHE_HOME"] = str(self.cache_dir)
        env["PADDLE_PDX_MODEL_SOURCE"] = settings.model_source
        env["PADDLEOCR_LANG"] = settings.language
        env["PADDLEOCR_VERSION"] = settings.ocr_version
        if settings.use_gpu is not None:
            env["PADDLEOCR_USE_GPU"] = "1" if settings.use_gpu else "0"
        if settings.cuda_bin_dir is not None:
            env["PADDLE_CUDA_BIN_DIR"] = str(settings.cuda_bin_dir)
        if settings.cudnn_bin_dir is not None:
            env["PADDLE_CUDNN_BIN_DIR"] = str(settings.cudnn_bin_dir)
        return env

    def _read_stdout(
        self,
        worker: subprocess.Popen[str],
        generation: int,
        output_queue: queue.Queue,
    ) -> None:
        try:
            assert worker.stdout is not None
            while True:
                line = worker.stdout.readline()
                output_queue.put((generation, line))
                if line == "":
                    return
        except (OSError, ValueError):
            output_queue.put((generation, ""))

    def _read_stderr(self, worker: subprocess.Popen[str], generation: int) -> None:
        try:
            assert worker.stderr is not None
            for line in worker.stderr:
                if generation != self._generation:
                    return
                stripped = line.strip()
                if stripped:
                    self._stderr_lines.append(stripped[:500])
        except (OSError, ValueError) as exc:
            logger.debug("PaddleOCR stderr reader stopped", error=str(exc))

    def _diagnostics(self, request_id: str = "") -> dict[str, Any]:
        phase = dict(self._last_phase)
        if request_id and phase.get("request_id") not in {"", request_id}:
            phase = {}
        return {
            "last_phase": str(phase.get("phase") or ""),
            "worker_stderr": list(self._stderr_lines)[-8:],
        }

    def _next_line(self, timeout_sec: float) -> str | None:
        if self._stdout_queue is None:
            return None
        deadline = time.monotonic() + timeout_sec
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                generation, line = self._stdout_queue.get(timeout=remaining)
            except queue.Empty:
                return None
            if generation == self._generation:
                return line

    def _start_worker(self):
        with self._lifecycle_lock:
            if self._worker and self._worker.poll() is None:
                return self._worker

            runner_script = Path(__file__).parent / "paddle_ocr_runner.py"
            ocr_python = self._resolve_python()
            logger.info(
                "Starting isolated PaddleOCR worker",
                script=str(runner_script),
                python=str(ocr_python),
            )
            self._generation += 1
            generation = self._generation
            self._stdout_queue = queue.Queue()
            self._stderr_lines = deque(maxlen=40)
            self._last_phase = {}
            self._worker = subprocess.Popen(
                [str(ocr_python), str(runner_script), "--worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
                bufsize=1,
                env=self._worker_env(),
            )
            threading.Thread(
                target=self._read_stdout,
                args=(self._worker, generation, self._stdout_queue),
                daemon=True,
            ).start()
            threading.Thread(
                target=self._read_stderr,
                args=(self._worker, generation),
                daemon=True,
            ).start()

            timeout = get_settings().ocr.worker_start_timeout_sec
            started = time.monotonic()
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    self._stop_worker_locked()
                    raise TimeoutError(
                        f"PaddleOCR worker startup timed out after {timeout:.1f}s"
                    )
                line = self._next_line(remaining)
                if line is None:
                    self._stop_worker_locked()
                    raise TimeoutError(
                        f"PaddleOCR worker startup timed out after {timeout:.1f}s"
                    )
                if line == "":
                    self._worker = None
                    raise RuntimeError("PaddleOCR worker exited during startup")
                stripped = line.strip()
                if stripped == "__OCR_WORKER_READY__":
                    self._request_count = 0
                    duration = time.monotonic() - started
                    logger.info(
                        "PaddleOCR worker ready",
                        pid=self._worker.pid,
                        startup=f"{duration:.2f}s",
                    )
                    context = current_run_context()
                    if context is not None:
                        context.record_step(
                            "ocr_startup", duration, pid=self._worker.pid
                        )
                    return self._worker
                if stripped.startswith("__OCR_WORKER_FAILED__"):
                    diagnostics = self._diagnostics()
                    self._worker = None
                    detail = diagnostics["worker_stderr"]
                    raise RuntimeError(
                        "PaddleOCR worker failed during startup"
                        + (f": {detail[-1]}" if detail else "")
                    )

    def _stop_worker(self) -> None:
        with self._lifecycle_lock:
            self._stop_worker_locked()

    def _stop_worker_locked(self) -> None:
        worker = self._worker
        self._worker = None
        self._stdout_queue = None
        self._last_phase = {}
        self._request_count = 0
        if worker is None:
            return
        try:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=2)
        except OSError as exc:
            logger.debug("PaddleOCR worker shutdown failed", error=str(exc))

    def _request(self, image_path: Path) -> list[dict[str, Any]]:
        attempts = get_settings().ocr.worker_max_attempts
        request_timeout = get_settings().ocr.request_timeout_sec
        for attempt in range(1, attempts + 1):
            worker = self._start_worker()
            try:
                return self._request_once(
                    worker,
                    image_path,
                    attempt=attempt,
                    timeout_sec=request_timeout,
                )
            except Exception as exc:
                logger.warning(
                    "PaddleOCR worker request failed",
                    attempt=attempt,
                    attempts=attempts,
                    error=str(exc),
                )
                self._stop_worker()
                if attempt >= attempts:
                    raise
        raise RuntimeError("PaddleOCR worker request failed")

    def _request_once(
        self,
        worker: subprocess.Popen[str],
        image_path: Path,
        *,
        attempt: int,
        timeout_sec: float,
    ) -> list[dict[str, Any]]:
        with observe_step(
            "ocr_request",
            attempt=attempt,
            request_timeout_sec=timeout_sec,
        ) as observation:
            observation.update(pid=worker.pid)
            try:
                request_id = self._send_request(worker, image_path)
                started = time.monotonic()
                results, timings = self._wait_for_result(
                    request_id,
                    timeout_sec=timeout_sec,
                    started=started,
                )
            except Exception as exc:
                observation.update(
                    success=False,
                    error=str(exc)[:300],
                    failure_code=(
                        "ocr_timeout"
                        if isinstance(exc, TimeoutError)
                        else "ocr_worker_error"
                    ),
                    last_phase=str(self._last_phase.get("phase") or ""),
                )
                raise

            self._request_count += 1
            duration = time.monotonic() - started
            logger.info(
                "PaddleOCR worker request completed",
                pid=worker.pid,
                attempt=attempt,
                request_count=self._request_count,
                duration=f"{duration:.2f}s",
                boxes=len(results),
                worker_timings=timings,
            )
            observation.update(
                request_count=self._request_count,
                boxes=len(results),
                success=True,
                **timings,
            )
            return results

    def _send_request(
        self,
        worker: subprocess.Popen[str],
        image_path: Path,
    ) -> str:
        if not worker.stdin or not worker.stdout:
            raise RuntimeError("PaddleOCR worker pipes are unavailable")
        request_id = f"{self._generation}:{time.time_ns()}"
        worker.stdin.write(
            json.dumps(
                {"request_id": request_id, "image_path": str(image_path)},
                ensure_ascii=False,
            )
            + "\n"
        )
        worker.stdin.flush()
        return request_id

    def _wait_for_result(
        self,
        request_id: str,
        *,
        timeout_sec: float,
        started: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        while True:
            remaining = timeout_sec - (time.monotonic() - started)
            if remaining <= 0:
                raise self._timeout_error(request_id, timeout_sec)
            line = self._next_line(remaining)
            if line is None:
                raise self._timeout_error(request_id, timeout_sec)
            if line == "":
                raise RuntimeError("PaddleOCR worker exited before returning a result")
            result = self._parse_worker_line(line.strip(), request_id)
            if result is not None:
                return result

    def _parse_worker_line(
        self,
        stripped: str,
        request_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        """작업자 이벤트는 상태에 반영하고 요청 결과만 반환한다."""

        if stripped.startswith("__OCR_EVENT__"):
            event = json.loads(
                stripped.removeprefix("__OCR_EVENT__").strip() or "{}"
            )
            if str(event.get("request_id") or "") == request_id:
                self._last_phase = event
            return None
        if stripped.startswith("__OCR_WORKER_ERROR__"):
            error = json.loads(
                stripped.removeprefix("__OCR_WORKER_ERROR__").strip() or "{}"
            )
            if str(error.get("request_id") or "") == request_id:
                raise RuntimeError(
                    "PaddleOCR worker error "
                    f"during {error.get('phase') or 'unknown'}: "
                    f"{error.get('error_type') or 'Error'}: "
                    f"{error.get('error') or ''}"
                )
            return None
        if stripped.startswith("__OCR_JSON_RESULT__"):
            decoded = json.loads(
                stripped.removeprefix("__OCR_JSON_RESULT__").strip() or "{}"
            )
            if str(decoded.get("request_id") or "") == request_id:
                return (
                    list(decoded.get("results") or []),
                    dict(decoded.get("timings") or {}),
                )
        return None

    def _timeout_error(self, request_id: str, timeout_sec: float) -> TimeoutError:
        diagnostics = self._diagnostics(request_id)
        return TimeoutError(
            "PaddleOCR worker request timed out after "
            f"{timeout_sec:.1f}s "
            f"(last_phase={diagnostics['last_phase'] or 'unknown'}, "
            f"stderr={diagnostics['worker_stderr']})"
        )

    @staticmethod
    def normalize_results(
        results: list[dict[str, Any]],
        *,
        scale: float = 1.0,
    ) -> list[dict[str, Any]]:
        boxes: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                raise TypeError("PaddleOCR worker result must be a mapping")
            confidence = float(item.get("confidence", item.get("conf", 0.0)))
            if confidence < 0.2:
                continue
            boxes.append(
                {
                    "bbox": [float(coord) / scale for coord in item["bbox"]],
                    "type": "text",
                    "text": str(item.get("text", "")),
                    "conf": confidence,
                }
            )
        logger.debug("PaddleOCR text detection complete", count=len(boxes))
        return boxes

    @staticmethod
    def scale_for_image(width: int, height: int) -> float:
        max_dim = get_settings().ocr.max_image_dim
        longest = max(width, height)
        if max_dim <= 0 or longest <= max_dim:
            return 1.0
        return max_dim / longest

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        width, height = image.size
        scale = self.scale_for_image(width, height)
        inference_image = image
        if scale != 1.0:
            inference_image = image.resize(
                (int(width * scale), int(height * scale)),
                Image.Resampling.BILINEAR,
            )
            logger.info(
                "Resized image for OCR inference",
                scale=round(scale, 3),
                original=(width, height),
                target=inference_image.size,
            )

        fd, raw_temp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        temp_path = Path(raw_temp)
        try:
            inference_image.convert("RGB").save(temp_path, "JPEG", quality=90)
            return self.normalize_results(self._request(temp_path), scale=scale)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Failed to remove temporary OCR image", error=str(exc))


__all__ = ["PaddleOcr"]
