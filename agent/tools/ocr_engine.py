"""PaddleOCR 문자와 OmniParser 아이콘을 하나의 SoM 결과로 합친다."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from agent.tools.omni_parser import OmniParser
from agent.tools.paddle_ocr import PaddleOcr
from agent.utils.logger import logger


def _area(box: list[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection_area(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def filter_overlaps(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """서로 거의 같은 검출 상자에서는 하나만 남긴다."""

    kept: list[dict[str, Any]] = []
    for box in sorted(boxes, key=lambda item: _area(item["bbox"]), reverse=True):
        bbox = box["bbox"]
        area = _area(bbox)
        if area <= 0:
            continue
        duplicate = False
        for existing in kept:
            intersection = _intersection_area(bbox, existing["bbox"])
            smaller_area = min(area, _area(existing["bbox"]))
            if smaller_area > 0 and intersection / smaller_area > 0.8:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    kept.sort(key=lambda item: (item["bbox"][1] // 20, item["bbox"][0]))
    return kept


def remove_text_containers(
    icon_boxes: list[dict[str, Any]],
    text_boxes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """OCR 글자를 감싼 의미 없는 아이콘 컨테이너를 제거한다."""

    filtered: list[dict[str, Any]] = []
    for icon in icon_boxes:
        icon_bbox = icon["bbox"]
        icon_area = _area(icon_bbox)
        contains_text = False
        for text in text_boxes:
            text_area = _area(text["bbox"])
            if text_area <= 0 or icon_area < text_area:
                continue
            if _intersection_area(icon_bbox, text["bbox"]) / text_area > 0.8:
                contains_text = True
                break
        if not contains_text:
            filtered.append(icon)
    return filtered


def _detect_local_icons(
    omni: OmniParser,
    region: Image.Image,
) -> list[dict[str, Any]]:
    """아이콘 크기를 유지한 채 모델 입력 여백을 붙여 국소 검출한다."""

    canvas_size = max(OmniParser.local_canvas_size, *region.size)
    canvas = Image.new("RGB", (canvas_size, canvas_size), "white")
    offset_x = (canvas_size - region.width) // 2
    offset_y = (canvas_size - region.height) // 2
    canvas.paste(region, (offset_x, offset_y))

    detected: list[dict[str, Any]] = []
    for element in omni.detect(canvas):
        x1, y1, x2, y2 = [float(value) for value in element["bbox"]]
        local_bbox = [
            x1 - offset_x,
            y1 - offset_y,
            x2 - offset_x,
            y2 - offset_y,
        ]
        center_x = (local_bbox[0] + local_bbox[2]) / 2
        center_y = (local_bbox[1] + local_bbox[3]) / 2
        if not (0 <= center_x <= region.width and 0 <= center_y <= region.height):
            continue
        clipped = [
            max(0.0, local_bbox[0]),
            max(0.0, local_bbox[1]),
            min(float(region.width), local_bbox[2]),
            min(float(region.height), local_bbox[3]),
        ]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        detected.append({**element, "bbox": clipped})
    return detected


def _draw_markers(
    image: Image.Image,
    elements: list[dict[str, Any]],
) -> tuple[Image.Image, dict[int, list[int]]]:
    marked_image = image.copy()
    draw = ImageDraw.Draw(marked_image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    marker_bboxes: dict[int, list[int]] = {}
    for marker_id, element in enumerate(elements):
        xmin, ymin, xmax, ymax = (int(value) for value in element["bbox"])
        marker_bboxes[marker_id] = [xmin, ymin, xmax, ymax]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 127, 80), width=2)

        label = f"[{marker_id}]"
        left, top, right, bottom = font.getbbox(label)
        label_width = right - left
        label_height = bottom - top
        label_top = max(0, ymin - label_height - 4)
        draw.rectangle(
            [xmin, label_top, xmin + label_width + 6, label_top + label_height + 4],
            fill=(0, 0, 0),
        )
        draw.text(
            (xmin + 3, label_top + 1),
            label,
            fill=(255, 255, 255),
            font=font,
        )
    return marked_image, marker_bboxes


class OcrEngine:
    """문자·아이콘 검출 결과를 물리 좌표 기반 마커로 합성한다."""

    def __init__(
        self,
        paddle: PaddleOcr | None = None,
        omni: OmniParser | None = None,
    ):
        self.paddle = paddle or PaddleOcr()
        self.omni = omni or OmniParser()
        logger.info("OCR engine initialized")

    def close(self) -> None:
        self.paddle.close()

    @property
    def worker_pid(self) -> int | None:
        return self.paddle.worker_pid

    def ensure_ready(self) -> None:
        self.paddle.ensure_ready()

    def process_image(
        self,
        image_path: Path,
        output_filename: str = "marked_screen.png",
        *,
        content_top: int = 0,
    ) -> tuple[Path, dict[int, list[int]], list[dict[str, Any]]]:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        content_top = max(0, min(int(content_top), image.height - 1))
        analysis_image = (
            image.crop((0, content_top, image.width, image.height))
            if content_top
            else image
        )

        ocr_started = time.perf_counter()
        text_boxes = self.paddle.detect(analysis_image)
        ocr_duration = time.perf_counter() - ocr_started

        omni_started = time.perf_counter()
        icon_boxes = filter_overlaps(self.omni.detect(analysis_image))
        icon_boxes = remove_text_containers(icon_boxes, text_boxes)
        omni_duration = time.perf_counter() - omni_started

        elements = text_boxes + icon_boxes
        elements.sort(key=lambda item: (item["bbox"][1] // 20, item["bbox"][0]))
        marked_image, marker_bboxes = _draw_markers(analysis_image, elements)
        if content_top:
            for bbox in marker_bboxes.values():
                bbox[1] += content_top
                bbox[3] += content_top
            for element in elements:
                element["bbox"][1] += content_top
                element["bbox"][3] += content_top
        output_path = image_path.parent / output_filename
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            marked_image.save(output_path, "JPEG", quality=85)
        else:
            marked_image.save(output_path)

        logger.info(
            "OCR analysis completed",
            output_path=str(output_path),
            markers_count=len(marker_bboxes),
            paddle_duration_sec=round(ocr_duration, 6),
            omni_duration_sec=round(omni_duration, 6),
            text_markers=len(text_boxes),
            icon_markers=len(icon_boxes),
            content_top=content_top,
        )
        return output_path, marker_bboxes, elements


__all__ = ["OcrEngine", "filter_overlaps", "remove_text_containers"]
