"""연속 화면 프레임 비교로 렌더링 변화와 안정화를 기다린다."""

import time
import os
from typing import Any, Optional

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


class WaitStable:
    """
    OCR 캡처 직전의 큰 화면 흔들림이 잦아들 때까지 기다리는 보조 모듈입니다.

    OpenCV 연속 프레임 비교로 행동 후 변화 시작과 렌더링 안정화를 판단합니다.
    pHash는 이 모듈의 일반 대기 시간이 아니라 저장 상태를 찾는 경우에만 사용합니다.
    """

    def __init__(self, perception_engine: Any):
        self.perception = perception_engine
        self.last_wait_result: dict = {}

    def _capture_memory_frame(
        self,
        region: Optional[dict] = None,
        sample_width: int = 360,
    ) -> np.ndarray:
        """현재 브라우저 화면을 파일 저장 없이 OpenCV 회색 프레임으로 가져옵니다."""

        if region is None:
            region = self.perception._get_browser_region()

        try:
            if region:
                sct_img = self.perception.sct.grab(region)
            else:
                sct_img = self.perception.sct.grab(self.perception.sct.monitors[1])

            raw = np.frombuffer(sct_img.bgra, dtype=np.uint8)
            frame = gray_frame(
                raw.reshape(
                    int(sct_img.height),
                    int(sct_img.width),
                    4,
                )
            )
            if sample_width > 0 and frame.shape[1] > sample_width:
                ratio = sample_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (
                        sample_width,
                        max(1, int(frame.shape[0] * ratio)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            return frame
        except Exception as e:
            logger.exception(
                "Failed to capture memory image for stabilization check", error=str(e)
            )
            raise

    def wait_for_change(
        self,
        reference_image_path: str,
        *,
        max_wait_sec: Optional[float] = None,
        check_interval_sec: Optional[float] = None,
        region: Optional[dict] = None,
    ) -> bool:
        """이전 스크린샷과 의미 있는 픽셀 차이가 생길 때까지만 짧게 기다립니다."""
        if not reference_image_path or not os.path.exists(reference_image_path):
            return False
        if max_wait_sec is None:
            max_wait_sec = get_settings().vision.transition_change_max_wait_sec
        if check_interval_sec is None:
            check_interval_sec = get_settings().vision.transition_change_check_sec
        minimum_ratio = get_settings().reflex.visual_change_min_ratio
        intensity_threshold = get_settings().reflex.visual_change_pixel_threshold
        try:
            reference = load_gray_frame(reference_image_path)
        except (OSError, ValueError) as exc:
            logger.debug(
                "Transition reference image could not be loaded", error=str(exc)
            )
            return False

        started = time.perf_counter()
        while (time.perf_counter() - started) < max_wait_sec:
            current = self._capture_memory_frame(
                region=region,
                sample_width=0,
            )
            changed_ratio = changed_pixel_ratio(
                reference,
                current,
                intensity_threshold=intensity_threshold,
            )
            if changed_ratio >= minimum_ratio:
                logger.info(
                    "Transition screen change detected",
                    elapsed_sec=round(time.perf_counter() - started, 3),
                    changed_ratio=round(changed_ratio, 4),
                )
                return True
            time.sleep(max(0.0, check_interval_sec))

        logger.info(
            "Transition screen change wait expired",
            max_wait_sec=max_wait_sec,
        )
        return False

    def wait(
        self,
        max_wait_sec: Optional[float] = None,
        check_interval_sec: Optional[float] = None,
        threshold_percent: Optional[float] = None,
        required_stable_frames: Optional[int] = None,
        region: Optional[dict] = None,
    ) -> bool:
        """
        연속 프레임의 픽셀 변화율이 여러 번 안정 범위에 들어올 때까지 기다립니다.
        정보량 검사는 이후 capture_usable_screen에서 수행합니다.

        Args:
            max_wait_sec: 최대 대기 시간(초). 이 시간이 넘어가면 무한 대기를 멈추고 반환.
            check_interval_sec: 화면 변화를 체크하는 간격(초).
            threshold_percent: 안정화로 판단할 픽셀 변화 강도(%).

        Returns:
            안정화 도달 시 True, 시간 초과 시 False
        """
        if max_wait_sec is None:
            max_wait_sec = get_settings().vision.stable_max_wait_sec
        if check_interval_sec is None:
            check_interval_sec = get_settings().vision.stable_check_interval_sec
        if threshold_percent is None:
            threshold_percent = get_settings().vision.stable_threshold_percent
        if required_stable_frames is None:
            required_stable_frames = get_settings().vision.stable_required_frames
        required_stable_frames = max(1, int(required_stable_frames))
        sample_width = get_settings().vision.stable_sample_width

        logger.info("Waiting for screen to stabilize...")
        start_time = time.perf_counter()
        probe_count = 0
        stable_count = 0
        last_diff_percent: float | None = None
        previous_frame = self._capture_memory_frame(
            region=region,
            sample_width=sample_width,
        )

        while (time.perf_counter() - start_time) < max_wait_sec:
            time.sleep(check_interval_sec)
            current_frame = self._capture_memory_frame(
                region=region,
                sample_width=sample_width,
            )
            probe_count += 1

            diff_ratio = mean_difference_percent(
                previous_frame,
                current_frame,
            )
            last_diff_percent = diff_ratio

            if diff_ratio <= threshold_percent:
                stable_count += 1
                if stable_count >= required_stable_frames:
                    elapsed = time.perf_counter() - start_time
                    self.last_wait_result = {
                        "stable": True,
                        "reason": "consecutive_frames_stable",
                        "elapsed_sec": round(elapsed, 3),
                        "probe_count": probe_count,
                        "stable_frames": stable_count,
                        "diff_percent": round(diff_ratio, 3),
                    }
                    logger.info("Screen stabilized", **self.last_wait_result)
                    return True
            else:
                stable_count = 0
                logger.info(
                    "Screen still changing...",
                    diff_percent=round(diff_ratio, 3),
                )
            previous_frame = current_frame

        self.last_wait_result = {
            "stable": False,
            "reason": "stability_timeout",
            "elapsed_sec": round(time.perf_counter() - start_time, 3),
            "probe_count": probe_count,
            "stable_frames": stable_count,
            "diff_percent": (
                round(last_diff_percent, 3) if last_diff_percent is not None else None
            ),
        }
        logger.warning(
            "Screen stabilization timeout reached",
            max_wait_sec=max_wait_sec,
            **self.last_wait_result,
        )
        return False
