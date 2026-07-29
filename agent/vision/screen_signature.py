"""OCR/마커 기반 replay를 보조하는 화면 서명(screen signature) 유틸리티."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from agent.vision.marker_geometry import (
    center_ratio_from_bbox,
    marker_bbox,
    ratio_rect_to_pixels,
    roi_rect_around_bbox,
)


CAPTURE_CONTEXT_VERSION = "browser-capture-v1"


def _normalize_text(value: Any) -> str:
    try:
        from agent.recipe.text_utils import normalize_text

        return normalize_text(value)
    except Exception:
        return " ".join(str(value or "").split())


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
        key=lambda marker: (marker_bbox(marker)[1], marker_bbox(marker)[0], marker.get("id", 0)),
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


def build_capture_context(size: list[int] | tuple[int, int], content_top: int = 0) -> dict[str, Any]:
    """레시피 좌표가 만들어진 브라우저 캡처 환경을 기록한다."""

    try:
        width = max(0, int(size[0]))
        height = max(0, int(size[1]))
        top = max(0, int(content_top or 0))
    except (TypeError, ValueError, IndexError):
        return {}
    if width <= 0 or height <= 0:
        return {}
    return {
        "version": CAPTURE_CONTEXT_VERSION,
        "size": [width, height],
        "content_top": min(top, height),
    }


def compute_roi_signature(
    image_path: str | Path,
    crop_rect_ratio: list[float],
    *,
    algorithm: str = "roi-phash-dct64-v1",
    capture_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """저장된 ROI 비율 좌표로 현재 스크린샷을 잘라 pHash를 계산한다."""

    try:
        with Image.open(image_path) as img:
            size = img.size
            rect = ratio_rect_to_pixels(crop_rect_ratio, size)
            if rect == [0, 0, 0, 0]:
                return {}
            crop = img.crop(tuple(rect))
            signature = {
                "algorithm": algorithm,
                "phash": perceptual_hash_image(crop),
                "crop_rect_ratio": [round(float(v), 4) for v in crop_rect_ratio],
                "source_size": list(size),
                "crop_size": list(crop.size),
            }
            context = dict(capture_context or {}) or build_capture_context(size)
            if context:
                signature["capture_context"] = context
            return signature
    except Exception:
        return {}


def compute_target_roi_signature_from_image(
    image: Image.Image,
    bbox: list[int] | tuple[int, int, int, int],
    size: list[int] | tuple[int, int],
    *,
    capture_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """열린 이미지에서 타깃 중심의 작은 ROI 서명을 계산한다."""

    crop_rect_ratio = roi_rect_around_bbox(bbox, size)
    if not crop_rect_ratio:
        return {}
    rect = ratio_rect_to_pixels(crop_rect_ratio, image.size)
    if rect == [0, 0, 0, 0]:
        return {}
    crop = image.crop(tuple(rect))
    context = dict(capture_context or {}) or build_capture_context(image.size)
    signature = {
        "algorithm": "roi-phash-dct64-v2",
        "phash": perceptual_hash_image(crop),
        "crop_rect_ratio": [round(float(v), 4) for v in crop_rect_ratio],
        "source_size": list(image.size),
        "crop_size": list(crop.size),
        "target_center_ratio": center_ratio_from_bbox(bbox, size),
    }
    if context:
        signature["capture_context"] = context
    return signature


def compute_target_roi_signature(
    image_path: str | Path,
    bbox: list[int] | tuple[int, int, int, int],
    size: list[int] | tuple[int, int],
    *,
    capture_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """타깃 bbox에서 ROI를 자동 산출하고 해당 영역 pHash를 계산한다."""

    try:
        with Image.open(image_path) as image:
            return compute_target_roi_signature_from_image(
                image,
                bbox,
                size,
                capture_context=capture_context,
            )
    except Exception:
        return {}


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


def compact_screen_context_signature(
    screen_signature: dict[str, Any] | None,
) -> dict[str, Any]:
    """경험 기반 탐색에 필요한 화면 비교 필드만 보존한다."""

    signature = dict(screen_signature or {})
    phash = str(signature.get("phash") or "")
    size = signature.get("size")
    if not phash or not isinstance(size, (list, tuple)) or len(size) != 2:
        return {}
    out = {
        "algorithm": str(signature.get("algorithm") or "phash-dct64-v1"),
        "phash": phash,
        "size": [int(size[0]), int(size[1])],
    }
    capture_context = dict(signature.get("capture_context") or {})
    if capture_context:
        out["capture_context"] = capture_context
    return out
