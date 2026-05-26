"""
이미지 처리 유틸리티.
PIL 이미지 리사이즈 → JPEG 압축 → Base64 인코딩 파이프라인을
nodes.py(reasoning_node)와 perception.py(analyze_ui) 양쪽에서 공유합니다.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path


def image_to_base64_jpeg(
    path: Path,
    max_dim: int = 1024,
    quality: int = 75,
    fast: bool = False,
) -> str:
    """
    이미지 파일을 읽어 리사이즈 후 JPEG Base64 문자열로 반환합니다.

    Args:
        path:     읽을 이미지 파일 경로
        max_dim:  긴 변의 최대 픽셀 수 (기본값 1024)
        quality:  JPEG 압축 품질 0~95 (기본값 75)
        fast:     True → BILINEAR(속도 우선), False → LANCZOS(화질 우선)

    Returns:
        Base64로 인코딩된 JPEG 이미지 문자열
    """
    from PIL import Image

    resample = Image.Resampling.BILINEAR if fast else Image.Resampling.LANCZOS

    with Image.open(path) as img:
        w, h = img.size
        if w > max_dim or h > max_dim:
            ratio = max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), resample)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
