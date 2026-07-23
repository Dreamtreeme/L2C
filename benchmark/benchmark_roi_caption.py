"""동일한 아이콘 ROI로 로컬 Florence와 Gemini 캡션 비용을 비교한다."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

from PIL import Image


BOXES = {
    "menu": [325, 205, 375, 250],
    "search": [1480, 205, 1530, 255],
    "business": [1800, 205, 1850, 255],
}


def crops(path: Path) -> list[Image.Image]:
    with Image.open(path) as image:
        source = image.convert("RGB")
        return [source.crop(box).resize((128, 128)) for box in BOXES.values()]


def run_florence(images: list[Image.Image], model_dir: Path, repeats: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to("cuda").eval()
    load_sec = time.perf_counter() - started
    durations = []
    outputs = []
    for _ in range(repeats):
        started = time.perf_counter()
        outputs = []
        for image in images:
            batch = processor(text="<CAPTION>", images=image, return_tensors="pt")
            batch = {
                key: value.to("cuda", dtype=torch.float16) if key == "pixel_values" else value.to("cuda")
                for key, value in batch.items()
            }
            # Florence 원격 코드가 이미지 토큰을 결합한 뒤 기존 텍스트 mask를 확장하지 못한다.
            batch.pop("attention_mask", None)
            with torch.inference_mode():
                generated = model.generate(**batch, max_new_tokens=20, num_beams=1, do_sample=False)
            outputs.extend(processor.batch_decode(generated, skip_special_tokens=True))
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    return {"load_sec": load_sec, "durations_sec": durations, "captions": outputs, "api_cost_usd": 0.0}


def image_data_url(images: list[Image.Image]) -> str:
    sheet = Image.new("RGB", (128 * len(images), 152), "white")
    for index, image in enumerate(images):
        sheet.paste(image, (128 * index, 0))
    buffer = io.BytesIO()
    sheet.save(buffer, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def run_gemini(images: list[Image.Image], model_name: str, repeats: int) -> dict:
    from agent.application.model_clients import get_google_chat_model
    from langchain_core.messages import HumanMessage

    llm = get_google_chat_model(model_name)
    message = HumanMessage(content=[
        {"type": "text", "text": "세 아이콘을 왼쪽부터 짧게 설명하세요. JSON 배열만 반환하세요."},
        {"type": "image_url", "image_url": {"url": image_data_url(images)}},
    ])
    durations = []
    usage = []
    output = ""
    for _ in range(repeats):
        started = time.perf_counter()
        response = llm.invoke([message])
        durations.append(time.perf_counter() - started)
        output = response.content
        usage.append(dict(getattr(response, "usage_metadata", None) or {}))
    return {"durations_sec": durations, "output": output, "usage": usage}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    images = crops(args.image)
    results = {
        "florence": run_florence(images, Path("models/omniparser/icon_caption"), args.repeats),
        "gemini-3.5-flash-lite": run_gemini(images, "gemini-3.5-flash-lite", args.repeats),
        "gemini-3.6-flash": run_gemini(images, "gemini-3.6-flash", args.repeats),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
