"""Reflex 후보 ROI만 VLM에 보여 주고 의미가 맞는 마커를 선택한다."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any

from PIL import Image, ImageDraw


def _candidate_sheet(image_path: str, markers: list[dict[str, Any]]) -> str:
    tile_size = 128
    sheet = Image.new("RGB", (tile_size * len(markers), tile_size + 24), "white")
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        for index, marker in enumerate(markers):
            bbox = [int(value) for value in marker.get("bbox", [])]
            if len(bbox) != 4:
                continue
            crop = source.crop(bbox)
            crop.thumbnail((tile_size - 12, tile_size - 12), Image.Resampling.LANCZOS)
            x = index * tile_size + (tile_size - crop.width) // 2
            y = (tile_size - crop.height) // 2
            sheet.paste(crop, (x, y))
            ImageDraw.Draw(sheet).text((index * tile_size + 4, tile_size + 3), f"ID {marker.get('id')}", fill="black")
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def select_marker_by_roi_caption(
    image_path: str,
    markers: list[dict[str, Any]],
    target: dict[str, str],
) -> tuple[int | None, dict[str, Any]]:
    """후보 crop만 캡션하여 저장된 의미와 같은 UI 요소를 선택한다."""

    target_label = str(target.get("label") or "").strip()
    target_component = str(target.get("component") or "").strip()
    target_intent = str(target.get("intent") or "").strip()
    if not markers or not any((target_label, target_component, target_intent)) or not os.getenv("GEMINI_API_KEY"):
        return None, {"reason": "roi_caption_unavailable"}

    from langchain_core.messages import HumanMessage
    from agent.application.model_clients import get_google_chat_model

    image = _candidate_sheet(image_path, markers)
    ids = [int(marker.get("id")) for marker in markers]
    prompt = (
        "저장된 웹 행동 레시피의 클릭 대상을 현재 UI 아이콘 후보에서 찾으십시오. "
        f"대상 명칭: {target_label or '없음'}. "
        f"구성요소: {target_component or '없음'}. "
        f"행동 목적: {target_intent or '없음'}. "
        "명칭의 문자 일치가 아니라 시각적 의미와 기능, 행동 목적이 명확히 일치하는 후보 하나만 고르십시오. "
        "확실하지 않거나 해당 후보가 없으면 marker_id를 null로 반환하십시오. "
        f"허용 ID: {ids}. JSON만 반환: {{\"marker_id\": 정수 또는 null, \"caption\": \"짧은 설명\"}}"
    )
    llm = get_google_chat_model(
        os.getenv("REFLEX_ROI_CAPTION_MODEL", "gemini-3.1-flash-lite"),
        temperature=0.0,
    )
    from agent.application.run_context import invoke_with_metrics

    response = invoke_with_metrics(llm, [HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
    ])], "reflex_roi_caption")
    output = response.content
    if isinstance(output, list):
        output = "\n".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in output)
    match = re.search(r"\{.*\}", str(output), re.DOTALL)
    data = json.loads(match.group(0) if match else str(output))
    marker_id = data.get("marker_id")
    if marker_id is None or int(marker_id) not in ids:
        return None, {"reason": "roi_caption_no_match", "caption": str(data.get("caption") or "")}
    return int(marker_id), {
        "reason": "roi_caption_matched",
        "caption": str(data.get("caption") or ""),
        "candidate_ids": ids,
    }
