"""Ollama 소형 비전 모델의 ROI 아이콘 분류 성능을 비교한다."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BOXES = [
    ("menu", [325, 205, 375, 250]),
    ("search", [1480, 205, 1530, 255]),
    ("business", [1800, 205, 1850, 255]),
]


def candidate_sheet(path: Path) -> bytes:
    tile_size = 192
    sheet = Image.new("RGB", (tile_size * len(BOXES), tile_size), "white")
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    with Image.open(path) as source:
        source = source.convert("RGB")
        for index, (_name, bbox) in enumerate(BOXES):
            crop = source.crop(bbox)
            crop.thumbnail((160, 140), Image.Resampling.LANCZOS)
            x = index * tile_size + (tile_size - crop.width) // 2
            y = 42 + (140 - crop.height) // 2
            sheet.paste(crop, (x, y))
            draw = ImageDraw.Draw(sheet)
            draw.rectangle([index * tile_size, 0, (index + 1) * tile_size - 1, tile_size - 1], outline="red", width=3)
            draw.text((index * tile_size + 8, 5), f"ID {index + 1}", fill="black", font=font)
    buffer = io.BytesIO()
    sheet.save(buffer, "PNG")
    return buffer.getvalue()


def benchmark(model: str, image: bytes, repeats: int) -> dict:
    import ollama

    prompt = (
        "/no_think 이미지에는 ID 1, 2, 3 UI 아이콘 후보가 있다. 검색 아이콘 하나를 고르라. "
        "JSON만 답하라: {\"marker_id\": 정수 또는 null, \"caption\": \"짧은 설명\"}"
    )
    rows = []
    output = ""
    for _ in range(repeats):
        started = time.perf_counter()
        response = ollama.generate(
            model=model,
            prompt=prompt,
            images=[image],
            stream=False,
            keep_alive="10m",
            options={"temperature": 0, "num_predict": 100},
            think=False,
        )
        elapsed = time.perf_counter() - started
        output = response.response or getattr(response, "thinking", "")
        rows.append({
            "elapsed_sec": elapsed,
            "total_duration_sec": response.total_duration / 1e9,
            "load_duration_sec": response.load_duration / 1e9,
            "prompt_eval_count": response.prompt_eval_count,
            "eval_count": response.eval_count,
        })
    ollama.generate(model=model, prompt="", keep_alive=0)
    return {"model": model, "runs": rows, "output": output}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--models", nargs="*", default=["minicpm-v4.6:1b", "qwen3-vl:2b"])
    args = parser.parse_args()
    image = candidate_sheet(args.image)
    results = [benchmark(model, image, args.repeats) for model in args.models]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
