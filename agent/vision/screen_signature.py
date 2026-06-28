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
        img = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        values = list(img.getdata())
    average = sum(values) / max(1, len(values))
    bits = "".join("1" if value > average else "0" for value in values)
    return f"{int(bits, 2):016x}"


def perceptual_hash(image_path: str | Path) -> str:
    """64비트 pHash를 계산한다. numpy가 없으면 평균 해시로 안전하게 후퇴한다."""

    try:
        import numpy as np

        size = 32
        with Image.open(image_path) as img:
            img = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
            pixels = np.asarray(img, dtype=np.float32)
        basis = _dct_basis(size)
        dct = basis @ pixels @ basis.T
        low = dct[:8, :8].flatten()
        median = float(np.median(low[1:]))
        bits = "".join("1" if value > median else "0" for value in low)
        return f"{int(bits, 2):016x}"
    except Exception:
        return _average_hash(image_path)


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


def compute_phash_signature(image_path: str | Path) -> dict[str, Any]:
    """OCR 없이 스크린샷 pHash와 화면 크기만 계산한다."""
    size: tuple[int, int] | None = None
    phash = ""
    try:
        size = image_dimensions(image_path)
        phash = perceptual_hash(image_path)
    except Exception:
        size = None
    return {
        "algorithm": "phash-dct64-v1",
        "phash": phash,
        "size": list(size or [0, 0]),
        "marker_count": 0,
        "marker_count_bucket": marker_count_bucket(0),
        "anchors": [],
    }


def compute_screen_signature(image_path: str | Path, markers: list[dict[str, Any]]) -> dict[str, Any]:
    """스크린샷 pHash와 OCR 앵커를 결합한 화면 서명을 만든다."""

    marker_count = len(markers or [])
    signature = compute_phash_signature(image_path)
    signature["marker_count"] = marker_count
    signature["marker_count_bucket"] = marker_count_bucket(marker_count)
    signature["anchors"] = anchor_texts(markers)
    return signature
