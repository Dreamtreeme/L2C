import json
import os
import site
import sys
import time
from pathlib import Path
from typing import Any

_DLL_DIRECTORY_HANDLES = []
SUPPORTED_PADDLE_VERSION = "3.3.1"
SUPPORTED_PADDLEOCR_VERSION = "3.7.0"


def _prepend_process_path(path: Path) -> None:
    path_text = str(path)
    current = os.environ.get("PATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if path_text.lower() not in {part.lower() for part in parts}:
        os.environ["PATH"] = path_text + os.pathsep + current


def _configure_runtime_paths() -> None:
    candidate_sites = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        candidate_sites.append(user_site)

    candidate_paths = []
    for variable_name in ("PADDLE_CUDA_BIN_DIR", "PADDLE_CUDNN_BIN_DIR"):
        configured = os.getenv(variable_name)
        if configured:
            candidate_paths.append(Path(configured))

    for site_dir in candidate_sites:
        nvidia_dir = Path(site_dir) / "nvidia"
        candidate_paths.extend(
            (
                nvidia_dir / "cu13" / "bin" / "x86_64",
                nvidia_dir / "cudnn" / "bin",
            )
        )

    for path in candidate_paths:
        try:
            exists = path.exists()
        except OSError:
            exists = False
        if not exists:
            continue
        _prepend_process_path(path)
        if hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))


def _load_paddleocr_runtime():
    _configure_runtime_paths()
    import paddle
    import paddleocr

    _validate_paddle_version(paddle)
    _validate_paddleocr_version(paddleocr)
    return paddle, paddleocr.PaddleOCR


def _validate_paddle_version(paddle_module: Any) -> None:
    installed_version = str(getattr(paddle_module, "__version__", "") or "unknown")
    if installed_version != SUPPORTED_PADDLE_VERSION:
        raise RuntimeError(
            "PaddlePaddle GPU runtime version mismatch: "
            f"installed={installed_version}, required={SUPPORTED_PADDLE_VERSION}. "
            "Install the version declared in requirements-ocr.txt."
        )


def _validate_paddleocr_version(paddleocr_module: Any) -> None:
    installed_version = str(getattr(paddleocr_module, "__version__", "") or "unknown")
    if installed_version != SUPPORTED_PADDLEOCR_VERSION:
        raise RuntimeError(
            "PaddleOCR version mismatch: "
            f"installed={installed_version}, required={SUPPORTED_PADDLEOCR_VERSION}. "
            "Install the version declared in requirements-ocr.txt."
        )


def _should_use_gpu(paddle_module: Any) -> bool:
    raw = os.getenv("PADDLEOCR_USE_GPU")
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return paddle_module.is_compiled_with_cuda() and paddle_module.device.cuda.device_count() > 0


def build_ocr():
    paddle_module, paddle_ocr_class = _load_paddleocr_runtime()
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
    device = "gpu:0" if _should_use_gpu(paddle_module) else "cpu"
    return paddle_ocr_class(
        lang=os.getenv("PADDLEOCR_LANG", "korean"),
        ocr_version=os.getenv("PADDLEOCR_VERSION", "PP-OCRv5"),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device,
    )


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _result_payload(result: Any) -> dict[str, Any]:
    value = getattr(result, "json", result)
    if callable(value):
        value = value()
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported PaddleOCR result type: {type(value).__name__}")
    payload = value.get("res", value)
    if not isinstance(payload, dict):
        raise TypeError("PaddleOCR result payload must be a mapping")
    return payload


def _box_from_polygon(polygon: Any) -> list[float]:
    points = _as_list(polygon)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def extract_text_boxes_from_results(results: list) -> list[dict]:
    extracted = []
    for result in results:
        payload = _result_payload(result)
        texts = _as_list(payload.get("rec_texts"))
        scores = _as_list(payload.get("rec_scores"))
        boxes = _as_list(payload.get("rec_boxes"))
        polygons = _as_list(payload.get("rec_polys"))

        if len(texts) != len(scores):
            raise ValueError("PaddleOCR text and score counts differ")
        if boxes and len(boxes) != len(texts):
            raise ValueError("PaddleOCR text and box counts differ")
        if not boxes and len(polygons) != len(texts):
            raise ValueError("PaddleOCR text and polygon counts differ")

        for index, text in enumerate(texts):
            if boxes:
                bbox = [float(value) for value in boxes[index]]
            else:
                bbox = _box_from_polygon(polygons[index])
            extracted.append(
                {
                    "text": str(text),
                    "confidence": float(scores[index]),
                    "bbox": bbox,
                }
            )
    return extracted


def emit_worker_payload(prefix: str, payload: dict[str, Any]) -> None:
    print(f"{prefix} {json.dumps(payload, ensure_ascii=False)}", flush=True)


def emit_worker_event(request_id: str, phase: str, **details: Any) -> None:
    emit_worker_payload(
        "__OCR_EVENT__",
        {"request_id": request_id, "phase": phase, **details},
    )


def worker_main() -> None:
    try:
        ocr = build_ocr()
        print("__OCR_WORKER_READY__", flush=True)
    except Exception as exc:
        print("__OCR_WORKER_FAILED__", flush=True)
        sys.stderr.write(str(exc))
        return

    for line in sys.stdin:
        request_id = ""
        phase = "request_received"
        try:
            request = json.loads(line)
            request_id = str(request.get("request_id") or "")
            image_path = str(request["image_path"])
            emit_worker_event(request_id, phase)

            phase = "inference_started"
            inference_started = time.perf_counter()
            emit_worker_event(request_id, phase)
            raw_results = list(ocr.predict(image_path))
            inference_sec = time.perf_counter() - inference_started
            phase = "inference_completed"
            emit_worker_event(
                request_id,
                phase,
                duration_sec=round(inference_sec, 6),
            )

            postprocess_started = time.perf_counter()
            extracted = extract_text_boxes_from_results(raw_results)
            postprocess_sec = time.perf_counter() - postprocess_started
            serialization_started = time.perf_counter()
            serialized_results = json.dumps(extracted, ensure_ascii=False)
            serialization_sec = time.perf_counter() - serialization_started
            phase = "result_serialized"
            emit_worker_event(
                request_id,
                phase,
                boxes=len(extracted),
                duration_sec=round(serialization_sec, 6),
            )
            timings = {
                "inference_sec": round(inference_sec, 6),
                "postprocess_sec": round(postprocess_sec, 6),
                "serialization_sec": round(serialization_sec, 6),
            }
            print(
                "__OCR_JSON_RESULT__ {"
                f'"request_id":{json.dumps(request_id, ensure_ascii=False)},'
                f'"results":{serialized_results},'
                f'"timings":{json.dumps(timings, ensure_ascii=False)}'
                "}",
                flush=True,
            )
        except Exception as exc:
            error = {
                "request_id": request_id,
                "phase": phase,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            emit_worker_payload("__OCR_WORKER_ERROR__", error)
            sys.stderr.write(
                f"request_id={request_id} phase={phase} "
                f"{type(exc).__name__}: {exc}\n"
            )
            sys.stderr.flush()


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != "--worker":
        raise SystemExit("paddle_ocr_runner.py는 --worker 모드로만 실행할 수 있습니다.")
    worker_main()


if __name__ == "__main__":
    main()
