import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image, ImageDraw, ImageFont
import torch

from agent.utils.logger import logger

class SomEngine:
    """
    순수 비전 기반 Set-of-Marks (SoM) 엔진입니다.
    화면 내의 클릭 가능한 요소(아이콘, 버튼, 텍스트)를 YOLOv8과 EasyOCR로 검출하고,
    화면에 숫자 마커 라벨을 합성한 이미지와 해당 마커의 물리 좌표 매핑을 제공합니다.
    """

    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.model_dir = self.root_dir / "models" / "omniparser"
        self.model_path = self.model_dir / "icon_detect" / "model.pt"
        
        # 1. YOLOv8 모델 로딩 및 가중치 확인
        self._ensure_model_downloaded()
        
        from ultralytics import YOLO
        logger.info("Loading local YOLOv8 OmniParser model...", model_path=str(self.model_path))
        self.yolo_model = YOLO(str(self.model_path))
        
        # 2. EasyOCR 로딩 (한국어, 영어 지원)
        logger.info("Initializing EasyOCR with CUDA support...")
        import easyocr
        self.reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
        logger.info("SomEngine initialization complete.")

    def _ensure_model_downloaded(self):
        """
        OmniParser YOLOv8 가중치 파일이 로컬에 없으면 Hugging Face에서 자동 다운로드합니다.
        """
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

    def process_image(self, image_path: Path, output_filename: str = "marked_screen.png") -> Tuple[Path, Dict[int, List[int]], Dict[int, List[int]], List[Dict[str, Any]]]:
        """
        스크린샷 이미지를 분석하여 마킹된 이미지 파일과 좌표 테이블을 생성합니다.

        Args:
            image_path: 분석할 스크린샷 이미지 절대 경로
            output_filename: 생성될 마킹 이미지의 파일명

        Returns:
            Tuple(마킹 이미지 절대 경로, {마커_ID: [x_중심점, y_중심점]})
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        raw_boxes = []

        # PIL 이미지 로드 및 리사이징 (EasyOCR & YOLOv8의 효율성을 위한 최적화)
        try:
            img = Image.open(image_path)
        except Exception as load_err:
            logger.error("Failed to load image for processing", error=str(load_err))
            raise

        original_w, original_h = img.size
        max_dim = 1280
        if original_w > max_dim or original_h > max_dim:
            scale = max_dim / max(original_w, original_h)
            inference_w = int(original_w * scale)
            inference_h = int(original_h * scale)
            # BILINEAR로 빠르게 리사이징
            inference_img = img.resize((inference_w, inference_h), Image.Resampling.BILINEAR)
            logger.debug("Resized image for local detection", scale=scale, original=(original_w, original_h), target=(inference_w, inference_h))
        else:
            scale = 1.0
            inference_img = img

        # 1. EasyOCR 실행 (텍스트 영역 추출)
        try:
            import numpy as np
            img_gray = np.array(inference_img.convert('L'))
            ocr_results = self.reader.readtext(img_gray)
            for bbox, text, conf in ocr_results:
                if conf < 0.2:  # 낮은 신뢰도 패스
                    continue
                # Bounding Box의 각 좌표를 원래 스크린샷 크기로 복원
                xs = [p[0] / scale for p in bbox]
                ys = [p[1] / scale for p in bbox]
                xmin, ymin = min(xs), min(ys)
                xmax, ymax = max(xs), max(ys)
                
                raw_boxes.append({
                    "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                    "type": "text",
                    "text": text,
                    "conf": float(conf)
                })
            logger.debug("EasyOCR element detection complete", count=len(ocr_results))
        except Exception as e:
            import traceback
            logger.error("EasyOCR inference failed", error=str(e), traceback=traceback.format_exc())

        # 2. YOLOv8 실행 (아이콘/버튼 추출)
        try:
            # 파일 경로 대신 PIL Image 객체를 전달하여 불필요한 디스크 I/O 제거
            yolo_results = self.yolo_model(inference_img, conf=0.15, verbose=False)
            yolo_count = 0
            if yolo_results and len(yolo_results) > 0:
                for box in yolo_results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy().tolist()  # [xmin, ymin, xmax, ymax]
                    conf = float(box.conf.item())
                    
                    # YOLOv8 검출 좌표를 원래 스크린샷 크기로 복원
                    coords_scaled = [c / scale for c in coords]
                    
                    raw_boxes.append({
                        "bbox": coords_scaled,
                        "type": "icon",
                        "text": "icon",
                        "conf": conf
                    })
                    yolo_count += 1
            logger.debug("YOLOv8 element detection complete", count=yolo_count)
        except Exception as e:
            logger.error("YOLOv8 inference failed", error=str(e))

        width, height = original_w, original_h

        # 3. 비최대 억제 및 중복 병합 (NMS / Overlap Filter)
        # 영역 넓이 기준으로 내림차순 정렬 (큰 영역이 작은 중복 영역을 덮을 수 있게 함)
        sorted_boxes = sorted(raw_boxes, key=lambda b: self._get_area(b["bbox"]), reverse=True)
        final_elements = []
        
        for box in sorted_boxes:
            bbox = box["bbox"]
            area = self._get_area(bbox)
            if area <= 0:
                continue
                
            should_keep = True
            for kept_box in final_elements:
                k_bbox = kept_box["bbox"]
                inter = self._get_intersection_area(bbox, k_bbox)
                if inter > 0:
                    k_area = self._get_area(k_bbox)
                    smaller_area = min(area, k_area)
                    overlap_ratio = inter / smaller_area
                    # 겹치는 영역이 작은 박스 면적의 80% 이상인 경우 중복으로 취급하여 제외
                    if overlap_ratio > 0.8:
                        should_keep = False
                        break
            if should_keep:
                final_elements.append(box)
                
        # 마커 배치의 직관성을 높이기 위해 상단->하단, 좌측->우측 순서로 번호 재정렬
        # ymin(상단) 순서로 정렬하되, 미세 차이는 xmin 순서로 정렬
        final_elements = sorted(final_elements, key=lambda e: (e["bbox"][1] // 20, e["bbox"][0]))
        
        logger.info("Overlap filtering complete", before=len(raw_boxes), after=len(final_elements))

        # 4. 이미지 그리기 및 좌표 매핑 사전 작성
        marked_img = img.copy()
        draw = ImageDraw.Draw(marked_img)
        
        # 폰트 로드 (Windows arial.ttf 선호, 실패 시 디폴트 폰트)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()

        marker_coords = {}
        marker_bboxes = {}
        
        for marker_id, elem in enumerate(final_elements):
            bbox = elem["bbox"]
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # 마커 클릭 좌표: Bounding Box의 중심점
            x_center = int((xmin + xmax) / 2)
            y_center = int((ymin + ymax) / 2)
            marker_coords[marker_id] = [x_center, y_center]
            marker_bboxes[marker_id] = [xmin, ymin, xmax, ymax]
            
            # 4.1 경계상자 그리기 (네온 오렌지색 테두리)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 127, 80), width=2)
            
            # 4.2 숫자 라벨 태그 그리기
            label_text = f"[{marker_id}]"
            # Modern pillow 10.0+ font bbox size calculation
            left, top, right, bottom = font.getbbox(label_text)
            text_w = right - left
            text_h = bottom - top
            
            tag_xmin = xmin
            tag_ymin = max(0, ymin - text_h - 4)
            tag_xmax = xmin + text_w + 6
            tag_ymax = tag_ymin + text_h + 4
            
            # 번호 가시성을 위한 검은색 배경 박스 그리기
            draw.rectangle([tag_xmin, tag_ymin, tag_xmax, tag_ymax], fill=(0, 0, 0))
            # 흰색으로 숫자 그리기
            draw.text((tag_xmin + 3, tag_ymin + 1), label_text, fill=(255, 255, 255), font=font)

        # 마크가 완료된 스크린샷 이미지 저장
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
            markers_count=len(marker_coords)
        )
        return output_path, marker_coords, marker_bboxes, final_elements
