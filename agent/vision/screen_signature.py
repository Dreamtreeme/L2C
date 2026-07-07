"""OCR/마커 기반 replay를 보조하는 화면 서명(screen signature) 유틸리티."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image


def _normalize_text(value: Any) -> str:
    try:
        from agent.recipe.state_key import normalize_text

        return normalize_text(value)
    except Exception:
        return " ".join(str(value or "").split())


def _bbox(marker: dict[str, Any]) -> list[int]:
    raw = marker.get("bbox") or [0, 0, 0, 0]
    if not isinstance(raw, list) or len(raw) != 4:
        return [0, 0, 0, 0]
    return [int(v or 0) for v in raw]


def _round_ratio(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def image_dimensions(image_path: str | Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


@lru_cache(maxsize=4)
def _dct_basis(size: int):
    import numpy as np

    basis = np.empty((size, size), dtype=np.float32)
    factor = math.pi / (2 * size)
    for k in range(size):
        scale = math.sqrt(1 / size) if k == 0 else math.sqrt(2 / size)
        for i in range(size):
            basis[k, i] = scale * math.cos((2 * i + 1) * k * factor)
    return basis


def _average_hash(image_path: str | Path) -> str:
    with Image.open(image_path) as img:
        return _average_hash_image(img)


def _average_hash_image(img: Image.Image) -> str:
    img = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    values = list(img.getdata())
    average = sum(values) / max(1, len(values))
    bits = "".join("1" if value > average else "0" for value in values)
    return f"{int(bits, 2):016x}"


def perceptual_hash(image_path: str | Path) -> str:
    """64비트 pHash를 계산한다. numpy가 없으면 평균 해시로 안전하게 후퇴한다."""

    try:
        with Image.open(image_path) as img:
            return perceptual_hash_image(img)
    except Exception:
        return _average_hash(image_path)


def perceptual_hash_image(img: Image.Image) -> str:
    """PIL 이미지 객체에서 64비트 pHash를 계산한다."""

    try:
        import numpy as np

        size = 32
        img = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        pixels = np.asarray(img, dtype=np.float32)
        basis = _dct_basis(size)
        dct = basis @ pixels @ basis.T
        low = dct[:8, :8].flatten()
        median = float(np.median(low[1:]))
        bits = "".join("1" if value > median else "0" for value in low)
        return f"{int(bits, 2):016x}"
    except Exception:
        return _average_hash_image(img)


def hamming_distance(left: str, right: str) -> int | None:
    if not left or not right:
        return None
    try:
        return (int(str(left), 16) ^ int(str(right), 16)).bit_count()
    except Exception:
        return None


def marker_count_bucket(count: int) -> str:
    if count <= 25:
        return "0-25"
    if count <= 50:
        return "26-50"
    if count <= 100:
        return "51-100"
    if count <= 150:
        return "101-150"
    if count <= 250:
        return "151-250"
    return "251+"


def anchor_texts(markers: list[dict[str, Any]], limit: int = 36) -> list[str]:
    """화면 서명 비교용 OCR 앵커 텍스트를 화면 순서대로 뽑는다."""

    ordered = sorted(
        [marker for marker in markers or [] if isinstance(marker, dict)],
        key=lambda marker: (_bbox(marker)[1], _bbox(marker)[0], marker.get("id", 0)),
    )
    out: list[str] = []
    seen: set[str] = set()
    for marker in ordered:
        text = _normalize_text(marker.get("text"))
        key = text.casefold().replace(" ", "")
        if len(key) < 2 or key.isdigit() or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def bbox_to_ratio(bbox: list[int] | tuple[int, int, int, int], size: list[int] | tuple[int, int]) -> list[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return []
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return []
    width = max(1, int(size[0] or 0))
    height = max(1, int(size[1] or 0))
    x1, y1, x2, y2 = [float(v or 0) for v in bbox]
    return [
        _round_ratio(x1 / width),
        _round_ratio(y1 / height),
        _round_ratio(x2 / width),
        _round_ratio(y2 / height),
    ]


def center_ratio_from_bbox(
    bbox: list[int] | tuple[int, int, int, int],
    size: list[int] | tuple[int, int],
) -> list[float]:
    ratios = bbox_to_ratio(bbox, size)
    if len(ratios) != 4:
        return []
    return [round((ratios[0] + ratios[2]) / 2, 4), round((ratios[1] + ratios[3]) / 2, 4)]


def marker_center_ratio(marker: dict[str, Any], size: list[int] | tuple[int, int]) -> list[float]:
    return center_ratio_from_bbox(_bbox(marker), size)


def _rect_to_ratio(rect: list[float], size: list[int] | tuple[int, int]) -> list[float]:
    if len(rect) != 4 or len(size) != 2:
        return []
    width = max(1, int(size[0] or 0))
    height = max(1, int(size[1] or 0))
    return [
        _round_ratio(rect[0] / width),
        _round_ratio(rect[1] / height),
        _round_ratio(rect[2] / width),
        _round_ratio(rect[3] / height),
    ]


def _ratio_rect_to_pixels(rect_ratio: list[float], size: list[int] | tuple[int, int]) -> list[int]:
    if not isinstance(rect_ratio, list) or len(rect_ratio) != 4 or len(size) != 2:
        return [0, 0, 0, 0]
    width = max(1, int(size[0] or 0))
    height = max(1, int(size[1] or 0))
    x1 = int(round(max(0.0, min(1.0, float(rect_ratio[0]))) * width))
    y1 = int(round(max(0.0, min(1.0, float(rect_ratio[1]))) * height))
    x2 = int(round(max(0.0, min(1.0, float(rect_ratio[2]))) * width))
    y2 = int(round(max(0.0, min(1.0, float(rect_ratio[3]))) * height))
    if x2 <= x1 or y2 <= y1:
        return [0, 0, 0, 0]
    return [x1, y1, x2, y2]


def roi_rect_around_bbox(
    bbox: list[int] | tuple[int, int, int, int],
    size: list[int] | tuple[int, int],
    *,
    margin_scale: float = 1.0,
    min_width_ratio: float = 0.20,
    min_height_ratio: float = 0.06,
    max_width_ratio: float = 0.45,
    max_height_ratio: float = 0.14,
) -> list[float]:
    """타깃 bbox 주변의 안정적인 ROI를 화면 비율 좌표로 만든다."""

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or len(size) != 2:
        return []
    width = max(1, int(size[0] or 0))
    height = max(1, int(size[1] or 0))
    x1, y1, x2, y2 = [float(v or 0) for v in bbox]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    crop_w = min(width * max_width_ratio, max(box_w * (1 + margin_scale * 2), width * min_width_ratio))
    crop_h = min(height * max_height_ratio, max(box_h * (1 + margin_scale * 2), height * min_height_ratio))
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    left = cx - crop_w / 2
    top = cy - crop_h / 2
    right = left + crop_w
    bottom = top + crop_h
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(width), right)
    bottom = min(float(height), bottom)
    if right <= left or bottom <= top:
        return []
    return _rect_to_ratio([left, top, right, bottom], [width, height])


def compute_roi_signature(image_path: str | Path, crop_rect_ratio: list[float]) -> dict[str, Any]:
    """저장된 ROI 비율 좌표로 현재 스크린샷을 잘라 pHash를 계산한다."""

    try:
        with Image.open(image_path) as img:
            size = img.size
            rect = _ratio_rect_to_pixels(crop_rect_ratio, size)
            if rect == [0, 0, 0, 0]:
                return {}
            crop = img.crop(tuple(rect))
            return {
                "algorithm": "roi-phash-dct64-v1",
                "phash": perceptual_hash_image(crop),
                "crop_rect_ratio": [round(float(v), 4) for v in crop_rect_ratio],
                "source_size": list(size),
                "crop_size": list(crop.size),
            }
    except Exception:
        return {}


def compute_target_roi_signature(
    image_path: str | Path,
    bbox: list[int] | tuple[int, int, int, int],
    size: list[int] | tuple[int, int],
) -> dict[str, Any]:
    """타깃 bbox에서 ROI를 자동 산출하고 해당 영역 pHash를 계산한다."""

    crop_rect_ratio = roi_rect_around_bbox(bbox, size)
    if not crop_rect_ratio:
        return {}
    signature = compute_roi_signature(image_path, crop_rect_ratio)
    if signature:
        signature["target_center_ratio"] = center_ratio_from_bbox(bbox, size)
    return signature


def compute_screen_signature(image_path: str | Path, markers: list[dict[str, Any]]) -> dict[str, Any]:
    """스크린샷 pHash와 OCR 앵커를 결합한 화면 서명을 만든다."""

    size: tuple[int, int] | None = None
    phash = ""
    try:
        size = image_dimensions(image_path)
        phash = perceptual_hash(image_path)
    except Exception:
        size = None
    marker_count = len(markers or [])
    signature = {
        "algorithm": "phash-dct64-v1",
        "phash": phash,
        "size": list(size or [0, 0]),
        "marker_count": marker_count,
        "marker_count_bucket": marker_count_bucket(marker_count),
        "anchors": anchor_texts(markers),
    }
    return signature
