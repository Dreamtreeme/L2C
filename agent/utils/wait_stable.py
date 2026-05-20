import time
from typing import Optional

from PIL import Image, ImageChops, ImageStat

from agent.tools.perception import PerceptionEngine
from agent.utils.logger import logger


class WaitStable:
    """
    클릭이나 스크롤 동작 후 UI 렌더링(애니메이션, 로딩 등)이 완료될 때까지
    화면의 픽셀 변화를 감지하여 대기하는 안정화 모듈입니다.
    """

    def __init__(self, perception_engine: PerceptionEngine):
        self.perception = perception_engine

    def _capture_memory_image(self) -> Image.Image:
        """
        현재 화면(브라우저 영역)을 파일로 저장하지 않고 메모리(PIL Image)로 즉시 가져옵니다.
        """
        region = self.perception._get_browser_region()
        
        try:
            if region:
                sct_img = self.perception.sct.grab(region)
            else:
                sct_img = self.perception.sct.grab(self.perception.sct.monitors[1])
                
            # mss의 raw BGRA 바이트를 BGRX 디코더로 PIL Image로 고속 변환
            return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        except Exception as e:
            logger.exception("Failed to capture memory image for stabilization check", error=str(e))
            raise

    def wait(self, max_wait_sec: float = 5.0, check_interval_sec: float = 0.2, threshold_percent: float = 1.0) -> bool:
        """
        화면 렌더링이 완료될 때까지 대기합니다.
        연속된 두 프레임의 픽셀 변화율 평균이 threshold_percent 이하가 되면 안정화된 것으로 판단합니다.
        
        Args:
            max_wait_sec: 최대 대기 시간(초). 이 시간이 넘어가면 무한 대기를 멈추고 반환.
            check_interval_sec: 화면 변화를 체크하는 간격(초).
            threshold_percent: 안정화로 판단할 픽셀 변화 강도(%).
            
        Returns:
            안정화 도달 시 True, 시간 초과 시 False
        """
        logger.info("Waiting for screen to stabilize...")
        start_time = time.time()
        
        prev_img = self._capture_memory_image()
        
        while (time.time() - start_time) < max_wait_sec:
            time.sleep(check_interval_sec)
            curr_img = self._capture_memory_image()
            
            # 1. 두 이미지 간의 픽셀 차이 절댓값 이미지 생성
            diff = ImageChops.difference(prev_img, curr_img)
            
            # 2. 이미지 통계 계산
            stat = ImageStat.Stat(diff)
            
            # 3. R, G, B 각 채널의 평균 픽셀 차이값(0~255)의 총합을 퍼센트로 환산
            diff_ratio = (sum(stat.mean) / (3 * 255.0)) * 100.0
            
            if diff_ratio <= threshold_percent:
                elapsed = time.time() - start_time
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
