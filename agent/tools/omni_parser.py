"""OmniParser 아이콘 검출 경계."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image

from agent.config import get_settings
from agent.utils.logger import logger


class OmniParser:
    """OmniParser의 YOLO 모델로 화면의 아이콘 후보를 검출한다."""

    local_canvas_size = 640

    def __init__(self, root_dir: Path | None = None):
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent.parent
        config_dir = get_settings().ocr.yolo_config_dir
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

        self.model_dir = self.root_dir / "models" / "omniparser"
        self.model_path = self.model_dir / "icon_detect" / "model.pt"
        self._ensure_model_downloaded()

        from ultralytics import YOLO

        logger.info(
            "Loading local OmniParser icon model",
            model_path=str(self.model_path),
        )
        self.model = YOLO(str(self.model_path))

    def _ensure_model_downloaded(self) -> None:
        if self.model_path.exists():
            return
        self.model_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download

        logger.info("OmniParser weights not found; downloading model")
        try:
            downloaded_path = hf_hub_download(
                repo_id="microsoft/OmniParser-v2.0",
                filename="icon_detect/model.pt",
                local_dir=str(self.model_dir),
            )
            logger.info(
                "OmniParser weights downloaded",
                downloaded_path=downloaded_path,
            )
        except Exception as exc:
            logger.error("Failed to download OmniParser weights", error=str(exc))
            raise

    @staticmethod
    def scale_for_image(width: int, height: int) -> float:
        max_dim = get_settings().ocr.inference_max_dim
        longest = max(width, height)
        return 1.0 if longest <= max_dim else max_dim / longest

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        width, height = image.size
        scale = self.scale_for_image(width, height)
        inference_image = image
        if scale != 1.0:
            inference_image = image.resize(
                (int(width * scale), int(height * scale)),
                Image.Resampling.BILINEAR,
            )
            logger.debug(
                "Resized image for OmniParser inference",
                scale=scale,
                original=(width, height),
                target=inference_image.size,
            )

        boxes: list[dict[str, Any]] = []
        results = self.model(inference_image, conf=0.15, verbose=False)
        if results:
            for box in results[0].boxes:
                coords = box.xyxy[0].cpu().numpy().tolist()
                boxes.append(
                    {
                        "bbox": [coord / scale for coord in coords],
                        "type": "icon",
                        "text": "icon",
                        "conf": float(box.conf.item()),
                    }
                )
        logger.debug("OmniParser icon detection complete", count=len(boxes))
        return boxes


__all__ = ["OmniParser"]
