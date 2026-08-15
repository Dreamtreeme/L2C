"""OpenCV 기반의 저비용 연속 프레임 비교."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def gray_frame(value: Any) -> np.ndarray:
    """PIL 이미지나 배열을 OpenCV 회색 프레임으로 정규화한다."""

    if isinstance(value, Image.Image):
        array = np.asarray(value.convert("RGB"))
        return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    array = np.asarray(value)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.ndim != 3:
        raise ValueError("비교할 화면 프레임의 차원이 올바르지 않습니다.")
    channels = array.shape[2]
    if channels == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    if channels == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    raise ValueError("지원하지 않는 화면 프레임 채널 수입니다.")


def load_gray_frame(image_path: str | Path) -> np.ndarray:
    """Windows 경로에서도 안정적으로 이미지를 읽어 회색 프레임으로 만든다."""

    with Image.open(image_path) as image:
        return gray_frame(image)


def resize_frame(
    frame: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """서로 다른 캡처를 같은 크기로 축소한다."""

    width, height = target_size
    return cv2.resize(
        gray_frame(frame),
        (max(1, int(width)), max(1, int(height))),
        interpolation=cv2.INTER_AREA,
    )


def changed_pixel_ratio(
    left: np.ndarray,
    right: np.ndarray,
    *,
    intensity_threshold: int,
    target_size: tuple[int, int] = (196, 212),
) -> float:
    """임계 강도보다 크게 달라진 픽셀의 비율을 반환한다."""

    left_frame = resize_frame(left, target_size)
    right_frame = resize_frame(right, target_size)
    difference = cv2.absdiff(left_frame, right_frame)
    _, changed = cv2.threshold(
        difference,
        min(255, max(0, int(intensity_threshold))),
        255,
        cv2.THRESH_BINARY,
    )
    return float(cv2.countNonZero(changed)) / float(changed.size)


def crop_frame_ratio(
    frame: np.ndarray,
    region_ratio: list[float],
) -> np.ndarray:
    """정규화된 화면 비율로 같은 상호작용 영역을 자른다."""

    source = gray_frame(frame)
    if len(region_ratio) != 4:
        raise ValueError("영역 비율은 [x1, y1, x2, y2] 형식이어야 합니다.")
    height, width = source.shape[:2]
    left = max(0, min(width - 1, round(region_ratio[0] * width)))
    top = max(0, min(height - 1, round(region_ratio[1] * height)))
    right = max(left + 1, min(width, round(region_ratio[2] * width)))
    bottom = max(top + 1, min(height, round(region_ratio[3] * height)))
    return source[top:bottom, left:right]


def changed_region_ratio(
    left: np.ndarray,
    right: np.ndarray,
    region_ratio: list[float],
    *,
    intensity_threshold: int,
) -> float:
    """두 캡처의 같은 화면 영역에서 달라진 픽셀 비율을 반환한다."""

    return changed_pixel_ratio(
        crop_frame_ratio(left, region_ratio),
        crop_frame_ratio(right, region_ratio),
        intensity_threshold=intensity_threshold,
    )


def mean_difference_percent(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    """연속 프레임의 평균 밝기 차이를 백분율로 반환한다."""

    left_frame = gray_frame(left)
    right_frame = gray_frame(right)
    if left_frame.shape != right_frame.shape:
        right_frame = cv2.resize(
            right_frame,
            (left_frame.shape[1], left_frame.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    difference = cv2.absdiff(left_frame, right_frame)
    return float(cv2.mean(difference)[0]) / 255.0 * 100.0


__all__ = [
    "changed_pixel_ratio",
    "changed_region_ratio",
    "crop_frame_ratio",
    "gray_frame",
    "load_gray_frame",
    "mean_difference_percent",
    "resize_frame",
]
