import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from agent.utils.logger import logger


class SomEngine:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        yolo_config_dir = Path(os.getenv("YOLO_CONFIG_DIR", self.root_dir / ".cache" / "ultralytics"))
        yolo_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))

        self.easyocr_dir = Path(os.getenv("EASYOCR_MODULE_PATH", self.root_dir / ".cache" / "easyocr"))
        self.easyocr_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("EASYOCR_MODULE_PATH", str(self.easyocr_dir))

        self.model_dir = self.root_dir / "models" / "omniparser"
        self.model_path = self.model_dir / "icon_detect" / "model.pt"

        self._ensure_model_downloaded()

        from ultralytics import YOLO

        logger.info("Loading local YOLOv8 OmniParser model", model_path=str(self.model_path))
        self.yolo_model = YOLO(str(self.model_path))

        try:
            import easyocr
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("EasyOCR dependencies are not installed. Run `pip install -r requirements.txt`.") from exc

        use_gpu = torch.cuda.is_available()
        logger.info("Loading EasyOCR reader", gpu=use_gpu, cache_dir=str(self.easyocr_dir))
        self.ocr_reader = easyocr.Reader(
            ["ko", "en"],
            gpu=use_gpu,
            model_storage_directory=str(self.easyocr_dir),
            user_network_directory=str(self.easyocr_dir),
            verbose=False,
        )
        logger.info("SomEngine initialization complete")

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

    def _normalize_easyocr_results(self, ocr_results: List, scale: float = 1.0) -> List[Dict]:
        raw_boxes = []
        for bbox, text, confidence in ocr_results:
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

        logger.debug("EasyOCR element detection complete", count=len(raw_boxes))
        return raw_boxes

    def _run_easy_ocr(self, image_path: Path, scale: float = 1.0) -> List[Dict]:
        import numpy as np

        with Image.open(image_path) as img:
            image_array = np.array(img.convert("L"))
        return self._normalize_easyocr_results(self.ocr_reader.readtext(image_array), scale=scale)

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
        ocr_image_path = image_path
        if ocr_scale != 1.0:
            ocr_img = img.resize(
                (int(original_w * ocr_scale), int(original_h * ocr_scale)),
                Image.Resampling.BILINEAR,
            )
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                ocr_image_path = Path(tmp.name)
            ocr_img.save(ocr_image_path, "JPEG", quality=85)
            logger.info(
                "Resized image for OCR inference",
                scale=round(ocr_scale, 3),
                original=(original_w, original_h),
                target=ocr_img.size,
            )

        try:
            raw_boxes = self._run_easy_ocr(ocr_image_path, scale=ocr_scale)
        finally:
            if ocr_image_path != image_path:
                try:
                    ocr_image_path.unlink(missing_ok=True)
                except Exception:
                    pass

        raw_boxes += self._run_yolo(inference_img, yolo_scale)
        final_elements = self._filter_overlaps(raw_boxes)
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
