import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image, ImageDraw, ImageFont

from agent.utils.logger import logger

class SomEngine:
    """
    순수 비전 기반 Set-of-Marks (SoM) 엔진입니다.
    화면 내의 클릭 가능한 요소(아이콘, 버튼, 텍스트)를 YOLOv8과 PaddleOCR로 검출하고,
    화면에 숫자 마커 라벨을 합성한 이미지와 해당 마커의 물리 좌표 매핑을 제공합니다.
    """

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.model_dir = self.root_dir / "models" / "omniparser"
        self.model_path = self.model_dir / "icon_detect" / "model.pt"

        self._ensure_model_downloaded()

        from ultralytics import YOLO
        logger.info("Loading local YOLOv8 OmniParser model...", model_path=str(self.model_path))
        self.yolo_model = YOLO(str(self.model_path))

        logger.info("SomEngine will invoke PaddleOCR (GPU/Isolated Subprocess) for text detection.")
        logger.info("SomEngine initialization complete.")

    def _ensure_model_downloaded(self):
        """OmniParser YOLOv8 가중치 파일이 로컬에 없으면 Hugging Face에서 자동 다운로드합니다."""
        if not self.model_path.exists():
            logger.info("YOLOv8 weights not found locally. Triggering Hugging Face download...")
            os.makedirs(self.model_dir, exist_ok=True)
            from huggingface_hub import hf_hub_download
            try:
                downloaded_path = hf_hub_download(
                    repo_id="microsoft/OmniParser-v2.0",
                    filename="icon_detect/model.pt",
                    local_dir=str(self.model_dir)
                )
                logger.info("Model weights downloaded successfully.", downloaded_path=downloaded_path)
            except Exception as e:
                logger.error("Failed to download model weights from Hugging Face.", error=str(e))
                raise

    # ------------------------------------------------------------------ #
    #  기하 연산 헬퍼                                                       #
    # ------------------------------------------------------------------ #

    def _get_area(self, box: List[float]) -> float:
        return (box[2] - box[0]) * (box[3] - box[1])

    def _get_intersection_area(self, box1: List[float], box2: List[float]) -> float:
        x_left  = max(box1[0], box2[0])
        y_top   = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bot   = min(box1[3], box2[3])
        if x_right < x_left or y_bot < y_top:
            return 0.0
        return (x_right - x_left) * (y_bot - y_top)

    # ------------------------------------------------------------------ #
    #  단계별 private 메서드                                                #
    # ------------------------------------------------------------------ #

    def _run_paddle_ocr(self, image_path: Path) -> List[Dict]:
        """PaddleOCR 서브프로세스를 실행하여 텍스트 박스 목록을 반환합니다."""
        import subprocess, json, sys

        runner_script = Path(__file__).parent / "paddle_ocr_runner.py"
        logger.debug("Invoking PaddleOCR runner", script=str(runner_script), image=str(image_path))

        res = subprocess.run(
            [sys.executable, str(runner_script), str(image_path)],
            capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )

        raw_boxes = []
        output_str = res.stdout.strip()
        marker = "__OCR_JSON_START__"
        if marker in output_str:
            json_str = output_str.split(marker)[-1].strip()
            ocr_results = json.loads(json_str) if json_str else []
        else:
            logger.warning("OCR JSON marker not found in output", output=output_str)
            ocr_results = []

        for item in ocr_results:
            if item["confidence"] < 0.2:
                continue
            raw_boxes.append({
                "bbox": item["bbox"],
                "type": "text",
                "text": item["text"],
                "conf": item["confidence"],
            })

        logger.debug("PaddleOCR element detection complete", count=len(raw_boxes))
        return raw_boxes

    def _run_yolo(self, inference_img, scale: float) -> List[Dict]:
        """YOLOv8으로 아이콘/버튼을 검출하고 원본 이미지 좌표로 복원하여 반환합니다."""
        raw_boxes = []
        try:
            yolo_results = self.yolo_model(inference_img, conf=0.15, verbose=False)
            if yolo_results and len(yolo_results) > 0:
                for box in yolo_results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    conf   = float(box.conf.item())
                    # 추론용 리사이즈 좌표 → 원본 크기로 복원
                    coords_scaled = [c / scale for c in coords]
                    raw_boxes.append({
                        "bbox": coords_scaled,
                        "type": "icon",
                        "text": "icon",
                        "conf": conf,
                    })
            logger.debug("YOLOv8 element detection complete", count=len(raw_boxes))
        except Exception as e:
            logger.error("YOLOv8 inference failed", error=str(e))
        return raw_boxes

    def _filter_overlaps(self, raw_boxes: List[Dict]) -> List[Dict]:
        """
        작은 박스 기준 overlap 비율 > 80% 인 중복 박스를 제거합니다.
        넓이 내림차순 정렬 후 상단→하단, 좌→우 순서로 최종 정렬합니다.
        """
        sorted_boxes = sorted(raw_boxes, key=lambda b: self._get_area(b["bbox"]), reverse=True)
        final_elements = []

        for box in sorted_boxes:
            bbox = box["bbox"]
            area = self._get_area(bbox)
            if area <= 0:
                continue

            is_duplicate = False
            for kept in final_elements:
                inter = self._get_intersection_area(bbox, kept["bbox"])
                if inter > 0:
                    smaller_area = min(area, self._get_area(kept["bbox"]))
                    if inter / smaller_area > 0.8:
                        is_duplicate = True
                        break

            if not is_duplicate:
                final_elements.append(box)

        # 상단→하단, 좌→우 재정렬 (마커 번호 직관성)
        final_elements.sort(key=lambda e: (e["bbox"][1] // 20, e["bbox"][0]))
        logger.info("Overlap filtering complete", before=len(raw_boxes), after=len(final_elements))
        return final_elements

    def _draw_markers(
        self,
        img: Image.Image,
        final_elements: List[Dict],
    ) -> Tuple[Image.Image, Dict[int, List[int]], Dict[int, List[int]]]:
        """
        마킹 이미지를 합성하고 marker_coords / marker_bboxes 딕셔너리를 반환합니다.

        Returns:
            (marked_img, marker_coords, marker_bboxes)
        """
        marked_img = img.copy()
        draw = ImageDraw.Draw(marked_img)

        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        marker_coords: Dict[int, List[int]] = {}
        marker_bboxes: Dict[int, List[int]] = {}

        for marker_id, elem in enumerate(final_elements):
            bbox = elem["bbox"]
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            x_center = (xmin + xmax) // 2
            y_center = (ymin + ymax) // 2
            marker_coords[marker_id] = [x_center, y_center]
            marker_bboxes[marker_id] = [xmin, ymin, xmax, ymax]

            # 경계 박스 (네온 오렌지)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 127, 80), width=2)

            # 번호 태그
            label_text = f"[{marker_id}]"
            left, top, right, bottom = font.getbbox(label_text)
            text_w, text_h = right - left, bottom - top

            tag_xmin = xmin
            tag_ymin = max(0, ymin - text_h - 4)
            tag_xmax = xmin + text_w + 6
            tag_ymax = tag_ymin + text_h + 4

            draw.rectangle([tag_xmin, tag_ymin, tag_xmax, tag_ymax], fill=(0, 0, 0))
            draw.text((tag_xmin + 3, tag_ymin + 1), label_text, fill=(255, 255, 255), font=font)

        return marked_img, marker_coords, marker_bboxes

    # ------------------------------------------------------------------ #
    #  공개 인터페이스                                                      #
    # ------------------------------------------------------------------ #

    def process_image(
        self,
        image_path: Path,
        output_filename: str = "marked_screen.png",
    ) -> Tuple[Path, Dict[int, List[int]], Dict[int, List[int]], List[Dict[str, Any]]]:
        """
        스크린샷을 분석하여 마킹 이미지와 좌표 테이블을 반환합니다.

        Pipeline:
            1. PaddleOCR  → 텍스트 박스
            2. YOLOv8     → 아이콘/버튼 박스
            3. Overlap filter (NMS-variant)
            4. 마킹 이미지 합성

        Returns:
            (marked_image_path, marker_coords, marker_bboxes, final_elements)
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # 원본 이미지 로드
        try:
            img = Image.open(image_path)
        except Exception as load_err:
            logger.error("Failed to load image for processing", error=str(load_err))
            raise

        original_w, original_h = img.size

        # 추론용 리사이즈 (PaddleOCR & YOLO 효율 최적화)
        max_dim = 1280
        if original_w > max_dim or original_h > max_dim:
            scale = max_dim / max(original_w, original_h)
            inference_img = img.resize(
                (int(original_w * scale), int(original_h * scale)),
                Image.Resampling.BILINEAR,
            )
            logger.debug("Resized image for inference", scale=scale,
                         original=(original_w, original_h),
                         target=inference_img.size)
        else:
            scale = 1.0
            inference_img = img

        # 1 & 2. 검출
        raw_boxes  = self._run_paddle_ocr(image_path)
        raw_boxes += self._run_yolo(inference_img, scale)

        # 3. 중복 제거 & 정렬
        final_elements = self._filter_overlaps(raw_boxes)

        # 4. 마킹 이미지 합성
        marked_img, marker_coords, marker_bboxes = self._draw_markers(img, final_elements)

        # 저장
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
