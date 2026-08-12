"""메모리 프레임 비교로 브라우저 렌더링 완료를 기다린다."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from agent.config import get_settings
from agent.utils.logger import logger
from agent.vision.frame_compare import (
    changed_pixel_ratio,
    gray_frame,
    load_gray_frame,
    mean_difference_percent,
)


def detect_browser_content_top(frame: Any) -> int:
    """브라우저 도구 영역과 웹 본문 사이의 가장 강한 수평 경계를 찾는다."""

    settings = get_settings().vision
    gray = gray_frame(frame)
    max_y = min(settings.content_scan_max_y, max(0, gray.shape[0] - 400))
    min_y = settings.content_scan_min_y
    if max_y <= min_y:
        return 0

    sample = gray
    sample_width = settings.content_scan_width
    if sample.shape[1] > sample_width:
        sample = cv2.resize(
            sample,
            (sample_width, sample.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    row_means = sample[: max_y + 1].mean(axis=1)
    deltas = np.abs(np.diff(row_means))
    search = deltas[max(0, min_y - 1) : max_y]
    if search.size == 0:
        return 0
    best_index = int(np.argmax(search)) + max(0, min_y - 1)
    best_y = best_index + 1
    best_delta = float(deltas[best_index])
    # 기존 RGB 채널 합 기준값을 회색조 평균 기준으로 환산한다.
    minimum_delta = settings.content_min_row_delta / 3.0
    return best_y if best_delta >= minimum_delta else 0


def frame_quality(frame: Any) -> dict[str, Any]:
    """저정보 회색 화면인지 OpenCV 지표로 판단한다."""

    settings = get_settings().vision
    gray = gray_frame(frame)
    top_setting = settings.content_top.strip().lower()
    if top_setting in {"", "auto"}:
        content_top = detect_browser_content_top(gray)
    else:
        try:
            content_top = max(0, int(top_setting))
        except ValueError:
            content_top = detect_browser_content_top(gray)

    bottom = max(content_top + 1, gray.shape[0] - settings.content_bottom_ignore_px)
    content = gray[min(content_top, gray.shape[0] - 1) : bottom]
    sample_width = settings.loading_sample_width
    if content.shape[1] > sample_width:
        ratio = sample_width / content.shape[1]
        content = cv2.resize(
            content,
            (sample_width, max(1, int(content.shape[0] * ratio))),
            interpolation=cv2.INTER_AREA,
        )

    stddev = float(np.std(content))
    edges = cv2.Laplacian(content, cv2.CV_32F)
    edge_mean = float(np.mean(np.abs(edges)))
    structure_edges = cv2.Canny(content, 20, 60)
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
        structure_edges,
        connectivity=8,
    )
    compact_component_count = sum(
        1
        for _, _, width, height, area in component_stats[1:component_count]
        if 2 <= area <= 120 and width <= 40 and height <= 25
    )
    histogram = cv2.calcHist([content], [0], None, [256], [0, 256])
    dominant_ratio = float(histogram.max()) / float(max(1, content.size))
    low_information = (
        stddev <= settings.loading_blank_max_stddev
        and edge_mean <= settings.loading_blank_max_edge_mean
        and dominant_ratio >= settings.loading_blank_min_dominant_ratio
        and compact_component_count < settings.loading_min_content_components
    )
    return {
        "low_information": low_information,
        "reason": (
            "low_information_page" if low_information else "page_content_present"
        ),
        "content_top": content_top,
        "stddev": round(stddev, 3),
        "edge_mean": round(edge_mean, 3),
        "dominant_ratio": round(dominant_ratio, 4),
        "compact_component_count": compact_component_count,
    }


class LoadingWait:
    """화면이 바뀐 뒤 움직임이 멈추고 본문이 나타날 때까지 기다린다."""

    def __init__(self, perception_engine: Any):
        self.perception = perception_engine
        self.last_result: dict[str, Any] = {}

    def _capture_memory_frame(
        self,
        region: dict[str, int] | None = None,
    ) -> np.ndarray:
        if region is None:
            region = self.perception._get_browser_region()
        try:
            source = (
                self.perception.sct.grab(region)
                if region
                else self.perception.sct.grab(self.perception.sct.monitors[1])
            )
            raw = np.frombuffer(source.bgra, dtype=np.uint8).reshape(
                int(source.height), int(source.width), 4
            )
            return gray_frame(raw)
        except Exception as exc:
            logger.exception("Failed to capture loading probe", error=str(exc))
            raise

    @staticmethod
    def _comparison_frame(frame: np.ndarray, sample_width: int) -> np.ndarray:
        if sample_width <= 0 or frame.shape[1] <= sample_width:
            return frame
        ratio = sample_width / frame.shape[1]
        return cv2.resize(
            frame,
            (sample_width, max(1, int(frame.shape[0] * ratio))),
            interpolation=cv2.INTER_AREA,
        )

    def wait_for_change(
        self,
        reference_image_path: str,
        *,
        max_wait_sec: float | None = None,
        check_interval_sec: float | None = None,
        region: dict[str, int] | None = None,
    ) -> bool:
        """직전 화면과 의미 있는 픽셀 차이가 시작될 때까지만 기다린다."""

        if not reference_image_path or not os.path.exists(reference_image_path):
            return False
        settings = get_settings()
        max_wait_sec = (
            settings.vision.transition_change_max_wait_sec
            if max_wait_sec is None
            else max_wait_sec
        )
        check_interval_sec = (
            settings.vision.transition_change_check_sec
            if check_interval_sec is None
            else check_interval_sec
        )
        try:
            reference = load_gray_frame(Path(reference_image_path))
        except (OSError, ValueError) as exc:
            logger.debug("Transition reference could not be loaded", error=str(exc))
            return False

        started = time.perf_counter()
        while time.perf_counter() - started < max_wait_sec:
            current = self._capture_memory_frame(region=region)
            ratio = changed_pixel_ratio(
                reference,
                current,
                intensity_threshold=settings.reflex.visual_change_pixel_threshold,
            )
            if ratio >= settings.reflex.visual_change_min_ratio:
                logger.info(
                    "Transition screen change detected",
                    elapsed_sec=round(time.perf_counter() - started, 3),
                    changed_ratio=round(ratio, 4),
                )
                return True
            time.sleep(max(0.0, check_interval_sec))
        logger.info("Transition screen change wait expired", max_wait_sec=max_wait_sec)
        return False

    def wait_until_ready(
        self,
        *,
        region: dict[str, int] | None = None,
        max_wait_sec: float | None = None,
        check_interval_sec: float | None = None,
        threshold_percent: float | None = None,
        required_stable_frames: int | None = None,
    ) -> dict[str, Any]:
        """파일을 만들지 않고 안정성과 본문 정보량을 한 루프에서 확인한다."""

        settings = get_settings().vision
        max_wait_sec = (
            settings.loading_timeout_sec
            if max_wait_sec is None
            else max_wait_sec
        )
        check_interval_sec = (
            settings.loading_check_interval_sec
            if check_interval_sec is None
            else check_interval_sec
        )
        threshold_percent = (
            settings.loading_motion_threshold_percent
            if threshold_percent is None
            else threshold_percent
        )
        required = max(
            1,
            int(
                settings.loading_stable_frames
                if required_stable_frames is None
                else required_stable_frames
            ),
        )

        started = time.perf_counter()
        previous_source = self._capture_memory_frame(region=region)
        previous = self._comparison_frame(
            previous_source,
            settings.loading_sample_width,
        )
        quality = frame_quality(previous_source)
        probe_count = 0
        stable_count = 0
        last_diff: float | None = None

        while time.perf_counter() - started < max_wait_sec:
            time.sleep(max(0.0, check_interval_sec))
            current_source = self._capture_memory_frame(region=region)
            current = self._comparison_frame(
                current_source,
                settings.loading_sample_width,
            )
            probe_count += 1
            last_diff = mean_difference_percent(previous, current)
            quality = frame_quality(current_source)
            stable_count = stable_count + 1 if last_diff <= threshold_percent else 0
            if stable_count >= required and not quality["low_information"]:
                self.last_result = {
                    **quality,
                    "ready": True,
                    "stable": True,
                    "wait_reason": "screen_ready",
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                    "probe_count": probe_count,
                    "stable_frames": stable_count,
                    "diff_percent": round(last_diff, 3),
                }
                logger.info("Screen ready", **self.last_result)
                return dict(self.last_result)
            previous = current

        self.last_result = {
            **quality,
            "ready": False,
            "stable": stable_count >= required,
            "wait_reason": (
                "low_information_timeout"
                if quality["low_information"]
                else "stability_timeout"
            ),
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "probe_count": probe_count,
            "stable_frames": stable_count,
            "diff_percent": round(last_diff, 3) if last_diff is not None else None,
        }
        logger.warning("Screen readiness wait expired", **self.last_result)
        return dict(self.last_result)


__all__ = [
    "LoadingWait",
    "detect_browser_content_top",
    "frame_quality",
]
