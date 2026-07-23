from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


APP_VERSIONS = {
    "fastapi": "0.139.2",
    "langchain-core": "1.5.0",
    "langchain-google-genai": "4.3.1",
    "langgraph": "1.2.9",
    "langgraph-checkpoint-sqlite": "3.1.0",
    "langsmith": "0.10.9",
    "numpy": "2.5.1",
    "opencv-python": "5.0.0.93",
    "pydantic": "2.13.4",
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
    "ultralytics": "8.4.104",
}

OCR_VERSIONS = {
    "numpy": "2.3.5",
    "opencv-contrib-python": "4.10.0.84",
    "paddleocr": "3.7.0",
    "paddlepaddle-gpu": "3.3.1",
    "paddlex": "3.7.2",
}

OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)


def require_python_313() -> None:
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(
            f"Python 3.13이 필요합니다. 현재 버전: {sys.version.split()[0]}"
        )


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def require_versions(expected: dict[str, str]) -> None:
    mismatches = []
    for distribution, required in expected.items():
        installed = installed_version(distribution)
        if installed != required:
            mismatches.append(
                f"{distribution}: installed={installed or 'missing'}, required={required}"
            )
    if mismatches:
        raise RuntimeError("패키지 버전 불일치:\n" + "\n".join(mismatches))


def require_single_opencv(expected_distribution: str) -> None:
    installed = [name for name in OPENCV_DISTRIBUTIONS if installed_version(name)]
    if installed != [expected_distribution]:
        raise RuntimeError(
            "OpenCV 배포판은 한 환경에 하나만 설치해야 합니다. "
            f"installed={installed}, required={[expected_distribution]}"
        )


def check_app_runtime() -> None:
    require_versions(APP_VERSIONS)
    require_single_opencv("opencv-python")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch에서 CUDA GPU를 사용할 수 없습니다.")
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"PyTorch CUDA 13.0이 필요합니다: {torch.version.cuda}")

    device_name = torch.cuda.get_device_name(0)
    value = torch.ones((32, 32), device="cuda").sum().item()
    print(f"app_runtime=ok python={sys.version.split()[0]} gpu={device_name} sum={value}")


def check_ocr_runtime() -> None:
    require_versions(OCR_VERSIONS)
    require_single_opencv("opencv-contrib-python")

    from agent.tools.paddle_ocr_runner import _configure_runtime_paths

    _configure_runtime_paths()

    import paddle

    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError("PaddlePaddle에서 CUDA GPU를 사용할 수 없습니다.")
    if str(paddle.version.cuda()) != "13.0":
        raise RuntimeError(f"PaddlePaddle CUDA 13.0이 필요합니다: {paddle.version.cuda()}")
    if str(paddle.version.cudnn()) != "9.13.0":
        raise RuntimeError(f"PaddlePaddle cuDNN 9.13.0이 필요합니다: {paddle.version.cudnn()}")

    paddle.set_device("gpu:0")
    value = paddle.ones([32, 32]).sum().numpy().item()
    print(
        "ocr_runtime=ok "
        f"python={sys.version.split()[0]} cuda={paddle.version.cuda()} "
        f"cudnn={paddle.version.cudnn()} sum={value}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("app", "ocr"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_python_313()
    if args.profile == "app":
        check_app_runtime()
    else:
        check_ocr_runtime()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
