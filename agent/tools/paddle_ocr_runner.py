import sys
from unittest.mock import MagicMock

# Crucial hack: Mock torch to prevent albumentations from importing PyTorch
# and triggering WinError 127 DLL symbol collisions with PaddlePaddle-GPU.
sys.modules['torch'] = MagicMock()

import os
import json
import paddle
from paddleocr import PaddleOCR

# Monkeypatch Config.switch_ir_optim to False to achieve 0.35s startup and 0.45s inference!
original_switch = paddle.inference.Config.switch_ir_optim
paddle.inference.Config.switch_ir_optim = lambda self, val: original_switch(self, False)


def build_ocr() -> PaddleOCR:
    gpu = paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    return PaddleOCR(use_gpu=gpu, lang='korean', show_log=False)


def extract_text_boxes(ocr: PaddleOCR, image_path: str) -> list[dict]:
    results = ocr.ocr(image_path, cls=False)

    extracted = []
    if results and results[0]:
        for line in results[0]:
            bbox, (text, conf) = line
            extracted.append({
                "text": text,
                "confidence": float(conf),
                "bbox": [float(bbox[0][0]), float(bbox[0][1]), float(bbox[2][0]), float(bbox[2][1])]
            })
    return extracted


def worker_main():
    try:
        ocr = build_ocr()
        print("__OCR_WORKER_READY__", flush=True)
    except Exception as e:
        print("__OCR_WORKER_FAILED__", flush=True)
        sys.stderr.write(str(e))
        return

    for line in sys.stdin:
        try:
            request = json.loads(line)
            image_path = request["image_path"]
            extracted = extract_text_boxes(ocr, image_path)
            print(f"__OCR_JSON_RESULT__ {json.dumps(extracted, ensure_ascii=False)}", flush=True)
        except Exception as e:
            print("__OCR_JSON_RESULT__ []", flush=True)
            sys.stderr.write(str(e))

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        worker_main()
        return

    if len(sys.argv) < 2:
        print("__OCR_JSON_START__\n[]")
        return
        
    image_path = sys.argv[1]

    try:
        ocr = build_ocr()
        extracted = extract_text_boxes(ocr, image_path)
        print(f"__OCR_JSON_START__\n{json.dumps(extracted, ensure_ascii=False)}")
    except Exception as e:
        print("__OCR_JSON_START__\n[]")
        sys.stderr.write(str(e))

if __name__ == "__main__":
    main()
