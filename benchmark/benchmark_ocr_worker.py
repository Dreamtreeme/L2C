"""실제 E2E 화면 순서로 PaddleOCR worker 지연과 안정성을 비교한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def screenshots_from_log(log_path: Path) -> list[Path]:
    screenshots: list[Path] = []
    seen: set[str] = set()
    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "Screen captured successfully":
            continue
        path = Path(str(event.get("output_path") or ""))
        if not path.name.startswith("screen_") or not path.exists():
            continue
        key = str(path.resolve()).casefold()
        if key in seen:
            continue
        seen.add(key)
        screenshots.append(path)
    return screenshots


def prepare_images(screenshots: list[Path], target_dir: Path, max_dim: int) -> list[Path]:
    from agent.tools.perception import PerceptionEngine

    detector = object.__new__(PerceptionEngine)
    prepared: list[Path] = []
    for index, source_path in enumerate(screenshots):
        with Image.open(source_path) as source:
            image = source.convert("RGB")
            crop_top = detector._detect_browser_content_top(image)
            if crop_top > 0:
                image = image.crop((0, crop_top, image.width, image.height))
            longest = max(image.size)
            if max_dim > 0 and longest > max_dim:
                scale = max_dim / longest
                image = image.resize(
                    (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                    Image.Resampling.BILINEAR,
                )
            output_path = target_dir / f"{index:03d}_{source_path.stem}.jpg"
            image.save(output_path, "JPEG", quality=90)
            prepared.append(output_path)
    return prepared


def build_ocr_client():
    from agent.tools.som_engine import SomEngine

    engine = object.__new__(SomEngine)
    engine.root_dir = ROOT_DIR
    engine.paddlex_cache_dir = engine._resolve_paddlex_cache_dir()
    engine._ocr_worker = None
    engine._ocr_worker_stdout_queue = None
    engine._ocr_worker_stderr_lines = deque(maxlen=40)
    engine._ocr_worker_last_phase = {}
    engine._ocr_worker_last_result_timings = {}
    engine._ocr_worker_generation = 0
    engine._ocr_worker_request_count = 0
    engine._ocr_worker_lifecycle_lock = threading.RLock()
    return engine


def text_fingerprint(results: list[dict[str, Any]]) -> str:
    texts = [str(item.get("text") or "").strip() for item in results]
    joined = "\n".join(text for text in texts if text)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def selected_screenshots(screenshots: list[Path], indices: list[int] | None) -> list[Path]:
    if not indices:
        return screenshots
    invalid = [index for index in indices if index < 0 or index >= len(screenshots)]
    if invalid:
        raise ValueError(f"Invalid screenshot indices: {invalid}")
    return [screenshots[index] for index in indices]


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["PADDLEOCR_USE_GPU"] = "1" if args.backend == "gpu" else "0"
    os.environ["SOM_OCR_REQUEST_TIMEOUT_SEC"] = str(args.timeout)
    os.environ["SOM_OCR_WORKER_MAX_ATTEMPTS"] = "1"

    screenshots = selected_screenshots(screenshots_from_log(args.log), args.image_indices)
    if not screenshots:
        raise RuntimeError(f"No E2E screenshots found in {args.log}")

    samples: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="l2c-ocr-benchmark-") as temp_dir:
        prepared = prepare_images(screenshots, Path(temp_dir), args.max_dim)
        engine = build_ocr_client()
        try:
            startup_started = time.perf_counter()
            engine.ensure_ocr_worker_ready()
            startup_sec = time.perf_counter() - startup_started
            for loop_index in range(args.loops):
                for image_index, image_path in enumerate(prepared):
                    started = time.perf_counter()
                    sample: dict[str, Any] = {
                        "loop": loop_index + 1,
                        "image_index": image_index,
                        "source": str(screenshots[image_index]),
                    }
                    try:
                        results = engine._run_paddle_ocr_worker(image_path)
                        sample.update(
                            {
                                "status": "success",
                                "duration_sec": round(time.perf_counter() - started, 6),
                                "boxes": len(results),
                                "text_fingerprint": text_fingerprint(results),
                                "worker_timings": dict(engine._ocr_worker_last_result_timings),
                                "worker_generation": engine._ocr_worker_generation,
                            }
                        )
                        if args.include_texts:
                            sample["texts"] = [
                                str(item.get("text") or "").strip()
                                for item in results
                                if str(item.get("text") or "").strip()
                            ]
                    except Exception as exc:
                        sample.update(
                            {
                                "status": "error",
                                "duration_sec": round(time.perf_counter() - started, 6),
                                "error": str(exc),
                                "diagnostics": engine._ocr_worker_diagnostics(),
                                "worker_generation": engine._ocr_worker_generation,
                            }
                        )
                    samples.append(sample)
        finally:
            engine._stop_ocr_worker()

    successes = [sample for sample in samples if sample["status"] == "success"]
    durations = [float(sample["duration_sec"]) for sample in successes]
    inference_durations = [
        float((sample.get("worker_timings") or {}).get("inference_sec") or 0.0)
        for sample in successes
    ]
    errors = [sample for sample in samples if sample["status"] == "error"]
    fingerprints_by_image: dict[int, set[str]] = {}
    for sample in successes:
        fingerprints_by_image.setdefault(int(sample["image_index"]), set()).add(
            str(sample.get("text_fingerprint") or "")
        )
    return {
        "backend": args.backend,
        "source_log": str(args.log),
        "loops": args.loops,
        "image_count": len(screenshots),
        "startup_sec": round(startup_sec, 6),
        "summary": {
            "request_count": len(samples),
            "success_count": len(successes),
            "error_count": len(errors),
            "p50_sec": round(statistics.median(durations), 6) if durations else 0.0,
            "p95_sec": round(percentile(durations, 0.95), 6),
            "max_sec": round(max(durations), 6) if durations else 0.0,
            "inference_p50_sec": round(statistics.median(inference_durations), 6)
            if inference_durations
            else 0.0,
            "unstable_fingerprint_images": sum(
                1 for fingerprints in fingerprints_by_image.values() if len(fingerprints) > 1
            ),
        },
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--backend", choices=("gpu", "cpu"), required=True)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-dim", type=int, default=1152)
    parser.add_argument("--image-indices", type=int, nargs="*")
    parser.add_argument("--include-texts", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_benchmark(args)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    return 0 if result["summary"]["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
