import os
import json
import site
import subprocess
import sys
import tempfile
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
            if path.exists():
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

    def _start_ocr_worker(self):
        if self._ocr_worker and self._ocr_worker.poll() is None:
            return self._ocr_worker

        runner_script = Path(__file__).parent / "paddle_ocr_runner.py"
        logger.info("Starting isolated PaddleOCR worker", script=str(runner_script))
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
        while True:
            line = self._ocr_worker.stdout.readline()
            if line == "":
                stderr = ""
                if self._ocr_worker.stderr is not None:
                    stderr = self._ocr_worker.stderr.read()
                self._ocr_worker = None
                raise RuntimeError(f"PaddleOCR worker exited during startup: {stderr.strip()}")
            stripped = line.strip()
            if stripped == "__OCR_WORKER_READY__":
                return self._ocr_worker
            if stripped.startswith("__OCR_WORKER_FAILED__"):
                stderr = ""
                if self._ocr_worker.stderr is not None:
                    stderr = self._ocr_worker.stderr.read()
                self._ocr_worker = None
                raise RuntimeError(f"PaddleOCR worker failed during startup: {stderr.strip()}")

    def _stop_ocr_worker(self) -> None:
        worker = getattr(self, "_ocr_worker", None)
        self._ocr_worker = None
        if not worker:
            return
        try:
            if worker.poll() is None:
                worker.terminate()
        except Exception:
            pass

    def _run_paddle_ocr_worker(self, image_path: Path) -> List[Dict[str, Any]]:
        worker = self._start_ocr_worker()
        if not worker.stdin or not worker.stdout:
            raise RuntimeError("PaddleOCR worker pipes are unavailable")

        request = json.dumps({"image_path": str(image_path)}, ensure_ascii=False)
        worker.stdin.write(request + "\n")
        worker.stdin.flush()

        while True:
            line = worker.stdout.readline()
            if line == "":
                self._stop_ocr_worker()
                raise RuntimeError("PaddleOCR worker exited before returning a result")
            stripped = line.strip()
            if not stripped.startswith("__OCR_JSON_RESULT__"):
                continue
            payload = stripped.removeprefix("__OCR_JSON_RESULT__").strip()
            return json.loads(payload) if payload else []

    def _run_paddle_ocr_once(self, image_path: Path) -> List[Dict[str, Any]]:
        runner_script = Path(__file__).parent / "paddle_ocr_runner.py"
        result = subprocess.run(
            [sys.executable, str(runner_script), str(image_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=self._ocr_worker_env(),
            check=False,
        )
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
        max_dim = self._env_int("SOM_OCR_MAX_DIM", 1280)
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

        text_boxes = self._filter_overlaps(self._run_paddle_ocr(ocr_image, scale=ocr_scale))
        icon_boxes = self._filter_overlaps(self._run_yolo(inference_img, yolo_scale))
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
        return output_path, marker_coords, marker_bboxes, final_elements
