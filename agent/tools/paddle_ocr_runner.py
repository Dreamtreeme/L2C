import json
import os
import site
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


if __name__ == "__main__":
    sys.modules["torch"] = MagicMock()

_DLL_DIRECTORY_HANDLES = []


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

    for site_dir in candidate_sites:
        nvidia_dir = Path(site_dir) / "nvidia"
        for path in (
            nvidia_dir / "cudnn" / "bin",
            nvidia_dir / "cublas" / "bin",
            nvidia_dir / "cuda_runtime" / "bin",
            nvidia_dir / "cuda_nvrtc" / "bin",
        ):
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

    if os.getenv("PADDLEOCR_IR_OPTIM", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        try:
            original_switch_ir_optim = paddle.inference.Config.switch_ir_optim
            paddle.inference.Config.switch_ir_optim = (
                lambda self, value: original_switch_ir_optim(self, False)
            )
        except Exception:
            pass

    from paddleocr import PaddleOCR

    return paddle, PaddleOCR


def _paddleocr_model_dirs() -> dict[str, str]:
    base_dir = Path(os.getenv("PADDLE_OCR_BASE_DIR", str(Path.home() / ".paddleocr")))
    whl_dir = base_dir / "whl"
    dirs = {
        "det_model_dir": whl_dir / "det" / "ml" / "Multilingual_PP-OCRv3_det_infer",
        "rec_model_dir": whl_dir / "rec" / "korean" / "korean_PP-OCRv4_rec_infer",
        "cls_model_dir": whl_dir / "cls" / "ch_ppocr_mobile_v2.0_cls_infer",
    }
    if all(path.exists() for path in dirs.values()):
        return {key: str(path) for key, path in dirs.items()}
    return {}


def _should_use_gpu(paddle_module: Any) -> bool:
    raw = os.getenv("PADDLEOCR_USE_GPU")
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return (
        paddle_module.device.is_compiled_with_cuda()
        and paddle_module.device.cuda.device_count() > 0
    )


def build_ocr():
    paddle_module, paddle_ocr_class = _load_paddleocr_runtime()
    return paddle_ocr_class(
        use_angle_cls=False,
        lang=os.getenv("PADDLEOCR_LANG", "korean"),
        use_gpu=_should_use_gpu(paddle_module),
        show_log=False,
        **_paddleocr_model_dirs(),
    )


def _looks_like_paddleocr_line(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (list, tuple))
        and isinstance(value[1], (list, tuple))
        and len(value[1]) >= 2
        and isinstance(value[1][0], str)
    )


def _iter_paddleocr_lines(results: list) -> list:
    if not results:
        return []
    first = results[0]
    if _looks_like_paddleocr_line(first):
        return results
    lines = []
    for page in results:
        if not page:
            continue
        if _looks_like_paddleocr_line(page):
            lines.append(page)
        else:
            lines.extend(line for line in page if _looks_like_paddleocr_line(line))
    return lines


def extract_text_boxes(ocr: Any, image_path: str) -> list[dict]:
    results = ocr.ocr(image_path, cls=False)
    return extract_text_boxes_from_results(results)


def extract_text_boxes_from_results(results: list) -> list[dict]:
    extracted = []
    for line in _iter_paddleocr_lines(results):
        bbox, (text, confidence) = line
        extracted.append(
            {
                "text": text,
                "confidence": float(confidence),
                "bbox": [
                    float(bbox[0][0]),
                    float(bbox[0][1]),
                    float(bbox[2][0]),
                    float(bbox[2][1]),
                ],
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
            raw_results = ocr.ocr(image_path, cls=False)
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
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        worker_main()
        return

    if len(sys.argv) < 2:
        print("__OCR_JSON_START__\n[]")
        return

    try:
        ocr = build_ocr()
        extracted = extract_text_boxes(ocr, sys.argv[1])
        print(f"__OCR_JSON_START__\n{json.dumps(extracted, ensure_ascii=False)}")
    except Exception as exc:
        print("__OCR_JSON_START__\n[]")
        sys.stderr.write(str(exc))


if __name__ == "__main__":
    main()
