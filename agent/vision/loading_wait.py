"""메모리 프레임 비교로 브라우저 렌더링 완료를 기다린다."""

from __future__ import annotations

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


def _wait_result(
    quality: dict[str, Any],
    *,
    ready: bool,
    wait_reason: str,
    elapsed: float,
    probe_count: int,
    stable_count: int,
    required_stable_frames: int,
    last_motion_elapsed: float,
    last_diff: float | None,
    visual_change_detected: bool,
    visual_change_ratio: float | None,
) -> dict[str, Any]:
    """로딩 대기 결과의 공통 관측값을 같은 형식으로 만든다."""

    return {
        **quality,
        "ready": ready,
        "stable": stable_count >= required_stable_frames,
        "wait_reason": wait_reason,
        "elapsed_sec": round(elapsed, 3),
        "probe_count": probe_count,
        "stable_frames": stable_count,
        "quiet_elapsed_sec": round(elapsed - last_motion_elapsed, 3),
        "diff_percent": round(last_diff, 3) if last_diff is not None else None,
        "visual_change_detected": visual_change_detected,
        "visual_change_ratio": (
            round(visual_change_ratio, 4)
            if visual_change_ratio is not None
            else None
        ),
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
            raw: np.ndarray = np.frombuffer(source.bgra, dtype=np.uint8).reshape(
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

    def wait_until_ready(
        self,
        *,
        reference_image_path: str | Path | None = None,
        region: dict[str, int] | None = None,
        max_wait_sec: float | None = None,
        change_grace_sec: float | None = None,
        check_interval_sec: float | None = None,
        quiet_period_sec: float | None = None,
        threshold_percent: float | None = None,
        required_stable_frames: int | None = None,
    ) -> dict[str, Any]:
        """화면 변화 시작, 로딩 종료와 안정화를 한 CV 루프에서 확인한다."""

        app_settings = get_settings()
        settings = app_settings.vision
        (
            max_wait_sec,
            change_grace_sec,
            check_interval_sec,
            quiet_period_sec,
            threshold_percent,
        ) = (
            settings.loading_timeout_sec if max_wait_sec is None else max_wait_sec,
            (
                settings.loading_change_grace_sec
                if change_grace_sec is None
                else change_grace_sec
            ),
            (
                settings.loading_check_interval_sec
                if check_interval_sec is None
                else check_interval_sec
            ),
            (
                settings.loading_quiet_period_sec
                if quiet_period_sec is None
                else max(0.0, quiet_period_sec)
            ),
            (
                settings.loading_motion_threshold_percent
                if threshold_percent is None
                else threshold_percent
            ),
        )
        required = max(
            1,
            int(
                settings.loading_stable_frames
                if required_stable_frames is None
                else required_stable_frames
            ),
        )

        reference: np.ndarray | None = None
        if reference_image_path:
            try:
                reference = load_gray_frame(Path(reference_image_path))
            except (OSError, ValueError) as exc:
                logger.debug("Loading reference could not be loaded", error=str(exc))

        started = time.perf_counter()
        previous_source = self._capture_memory_frame(region=region)
        previous = self._comparison_frame(
            previous_source,
            settings.loading_sample_width,
        )
        quality = frame_quality(previous_source)
        probe_count = 0
        stable_count = 0
        last_motion_elapsed = 0.0
        last_diff: float | None = None
        visual_change_ratio: float | None = None
        visual_change_detected = False
        if reference is not None:
            visual_change_ratio = changed_pixel_ratio(
                reference,
                previous_source,
                intensity_threshold=(
                    app_settings.reflex.visual_change_pixel_threshold
                ),
            )
            visual_change_detected = (
                visual_change_ratio
                >= app_settings.reflex.visual_change_min_ratio
            )

        elapsed = 0.0
        while elapsed < max_wait_sec:
            time.sleep(max(0.0, check_interval_sec))
            current_source = self._capture_memory_frame(region=region)
            current = self._comparison_frame(
                current_source,
                settings.loading_sample_width,
            )
            probe_count += 1
            elapsed = time.perf_counter() - started
            last_diff = mean_difference_percent(previous, current)
            quality = frame_quality(current_source)
            if last_diff <= threshold_percent:
                stable_count += 1
            else:
                stable_count = 0
                last_motion_elapsed = elapsed
            if reference is not None:
                visual_change_ratio = changed_pixel_ratio(
                    reference,
                    current_source,
                    intensity_threshold=(
                        app_settings.reflex.visual_change_pixel_threshold
                    ),
                )
                visual_change_detected = visual_change_detected or (
                    visual_change_ratio
                    >= app_settings.reflex.visual_change_min_ratio
                )
            change_wait_complete = (
                reference is None
                or visual_change_detected
                or elapsed >= max(0.0, change_grace_sec)
            )
            quiet_wait_complete = (
                reference is None
                or not visual_change_detected
                or elapsed - last_motion_elapsed >= quiet_period_sec
            )
            if (
                change_wait_complete
                and quiet_wait_complete
                and stable_count >= required
                and not quality["low_information"]
            ):
                self.last_result = _wait_result(
                    quality,
                    ready=True,
                    wait_reason="screen_ready",
                    elapsed=elapsed,
                    probe_count=probe_count,
                    stable_count=stable_count,
                    required_stable_frames=required,
                    last_motion_elapsed=last_motion_elapsed,
                    last_diff=last_diff,
                    visual_change_detected=visual_change_detected,
                    visual_change_ratio=visual_change_ratio,
                )
                logger.info("Screen ready", **self.last_result)
                return dict(self.last_result)
            previous = current

        self.last_result = _wait_result(
            quality,
            ready=False,
            wait_reason=(
                "low_information_timeout"
                if quality["low_information"]
                else "stability_timeout"
            ),
            elapsed=elapsed,
            probe_count=probe_count,
            stable_count=stable_count,
            required_stable_frames=required,
            last_motion_elapsed=last_motion_elapsed,
            last_diff=last_diff,
            visual_change_detected=visual_change_detected,
            visual_change_ratio=visual_change_ratio,
        )
        logger.warning("Screen readiness wait expired", **self.last_result)
        return dict(self.last_result)


__all__ = [
    "LoadingWait",
    "detect_browser_content_top",
    "frame_quality",
]
