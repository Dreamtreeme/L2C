from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def prepare_omniparser() -> None:
    from huggingface_hub import hf_hub_download

    model_dir = ROOT_DIR / "models" / "omniparser"
    model_path = model_dir / "icon_detect" / "model.pt"
    hf_hub_download(
        repo_id="microsoft/OmniParser-v2.0",
        filename="icon_detect/model.pt",
        local_dir=str(model_dir),
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"OmniParser 모델 다운로드에 실패했습니다: {model_path}")
    print(f"asset=omniparser status=ok path={model_path}")


def prepare_paddleocr() -> None:
    cache_dir = ROOT_DIR / ".cache" / "paddlex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")

    from agent.tools.paddle_ocr_runner import build_ocr

    ocr = build_ocr()
    del ocr
    gc.collect()
    print(f"asset=paddleocr status=ok cache={cache_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        choices=("omniparser", "paddleocr"),
        required=True,
    )
    args = parser.parse_args()

    if args.component == "omniparser":
        prepare_omniparser()
    else:
        prepare_paddleocr()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
