import time
import os
from typing import Optional

from PIL import Image, ImageChops, ImageStat

from agent.tools.perception import PerceptionEngine
from agent.utils.logger import logger


class WaitStable:
    """
    OCR 캡처 직전의 큰 화면 흔들림이 잦아들 때까지 기다리는 보조 모듈입니다.

    콘텐츠 로딩 완료 여부는 판단하지 않습니다. 정상 결과와 로딩 중 화면의 구분은
    perception 단계의 전환 계약(transition contract)이 담당합니다.
    """

    def __init__(self, perception_engine: PerceptionEngine):
        self.perception = perception_engine

    def _capture_memory_image(self, region: Optional[dict] = None, sample_width: int = 360) -> Image.Image:
        """
        현재 화면(브라우저 영역)을 파일로 저장하지 않고 메모리(PIL Image)로 즉시 가져옵니다.
        """
        if region is None:
            region = self.perception._get_browser_region()
        
        try:
            if region:
                sct_img = self.perception.sct.grab(region)
            else:
                sct_img = self.perception.sct.grab(self.perception.sct.monitors[1])
                
            # mss의 raw BGRA 바이트를 BGRX 디코더로 PIL Image로 고속 변환
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            if sample_width > 0 and img.width > sample_width:
                ratio = sample_width / img.width
                img = img.resize((sample_width, max(1, int(img.height * ratio))), Image.Resampling.BILINEAR)
            return img
        except Exception as e:
            logger.exception("Failed to capture memory image for stabilization check", error=str(e))
            raise

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return default

    def wait(
        self,
        max_wait_sec: Optional[float] = None,
        check_interval_sec: Optional[float] = None,
        threshold_percent: Optional[float] = None,
        region: Optional[dict] = None,
    ) -> bool:
        """
        연속된 두 프레임의 픽셀 변화율 평균이 threshold_percent 이하가 될 때까지 기다립니다.
        반환값은 캡처 안정화 여부일 뿐 페이지 준비 완료를 의미하지 않습니다.
        
        Args:
            max_wait_sec: 최대 대기 시간(초). 이 시간이 넘어가면 무한 대기를 멈추고 반환.
            check_interval_sec: 화면 변화를 체크하는 간격(초).
            threshold_percent: 안정화로 판단할 픽셀 변화 강도(%).
            
        Returns:
            안정화 도달 시 True, 시간 초과 시 False
        """
        if max_wait_sec is None:
            max_wait_sec = self._env_float("VISION_STABLE_MAX_WAIT_SEC", 2.0)
        if check_interval_sec is None:
            check_interval_sec = self._env_float("VISION_STABLE_CHECK_INTERVAL_SEC", 0.04)
        if threshold_percent is None:
            threshold_percent = self._env_float("VISION_STABLE_THRESHOLD_PERCENT", 1.0)
        sample_width = self._env_int("VISION_STABLE_SAMPLE_WIDTH", 360)

        logger.info("Waiting for screen to stabilize...")
        start_time = time.perf_counter()
        
        prev_img = self._capture_memory_image(region=region, sample_width=sample_width)
        
        while (time.perf_counter() - start_time) < max_wait_sec:
            time.sleep(check_interval_sec)
            curr_img = self._capture_memory_image(region=region, sample_width=sample_width)
            
            # 1. 두 이미지 간의 픽셀 차이 절댓값 이미지 생성
            diff = ImageChops.difference(prev_img, curr_img)
            
            # 2. 이미지 통계 계산
            stat = ImageStat.Stat(diff)
            
            # 3. R, G, B 각 채널의 평균 픽셀 차이값(0~255)의 총합을 퍼센트로 환산
            diff_ratio = (sum(stat.mean) / (3 * 255.0)) * 100.0
            
            if diff_ratio <= threshold_percent:
                elapsed = time.perf_counter() - start_time
                logger.info(
                    "Screen stabilized", 
                    elapsed_sec=round(elapsed, 2), 
                    diff_percent=round(diff_ratio, 3)
                )
                return True
                
            logger.info("Screen still changing...", diff_percent=round(diff_ratio, 3))
            prev_img = curr_img
            
        logger.warning("Screen stabilization timeout reached", max_wait_sec=max_wait_sec)
        return False
