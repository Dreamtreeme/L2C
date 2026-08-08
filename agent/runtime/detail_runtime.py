"""상세 페이지 OCR을 읽기용 줄과 누적 버퍼로 변환한다."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.runtime.worker_contracts import WorkerState

from agent.config import get_settings
from agent.runtime.site_context import is_job_detail_context, page_guidance_for_url
from agent.utils.logger import logger
from agent.vision.marker_geometry import marker_bbox


def marker_prompt_rank(marker: dict) -> tuple[int, int, int]:
    """의미를 추측하지 않고 화면 읽기 순서로 마커를 정렬한다."""

    bbox = marker.get("bbox", [0, 0, 0, 0])
    y = int(bbox[1]) if len(bbox) == 4 else 0
    x = int(bbox[0]) if len(bbox) == 4 else 0
    marker_id = int(marker.get("id") or 0)
    return (y, x, marker_id)


def is_icon_marker(marker: dict) -> bool:
    text = str(marker.get("text") or "")
    marker_type = str(marker.get("type") or "").strip().lower()
    return (
        marker_type == "icon"
        or text == "icon"
        or text.startswith("상호작용 가능한 요소 (")
        or text == "상호작용 가능한 요소"
    )


def line_bbox(markers: list[dict]) -> list[int]:
    boxes = [marker_bbox(marker) for marker in markers]
    boxes = [box for box in boxes if box != [0, 0, 0, 0]]
    if not boxes:
        return [0, 0, 0, 0]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def join_line_marker_text(markers: list[dict]) -> str:
    ordered = sorted(markers, key=lambda marker: (marker_bbox(marker)[0], marker_bbox(marker)[1]))
    pieces: list[str] = []
    for marker in ordered:
        text = str(marker.get("text") or "").strip()
        if text:
            pieces.append(text)
    joined = " ".join(pieces)
    joined = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", joined)
    joined = re.sub(r"([(\[\{])\s+", r"\1", joined)
    return re.sub(r"\s+", " ", joined).strip()


def group_text_markers_into_lines(markers: list[dict]) -> list[dict]:
    """가까운 y축의 OCR 마커를 읽기 순서의 문장 줄로 묶는다."""

    text_markers = [
        marker
        for marker in markers
        if str(marker.get("text") or "").strip() and not is_icon_marker(marker)
    ]
    if not text_markers:
        return []
    heights = sorted(max(1, marker_bbox(marker)[3] - marker_bbox(marker)[1]) for marker in text_markers)
    median_height = heights[len(heights) // 2] if heights else 16
    tolerance = max(8, min(24, int(median_height * 0.7)))
    lines: list[dict] = []
    ordered_markers = sorted(
        text_markers,
        key=lambda item: (
            (marker_bbox(item)[1] + marker_bbox(item)[3]) / 2,
            marker_bbox(item)[0],
        ),
    )
    for marker in ordered_markers:
        bbox = marker_bbox(marker)
        center_y = (bbox[1] + bbox[3]) / 2
        matched = None
        for line in lines:
            if abs(center_y - line["center_y"]) <= tolerance:
                matched = line
                break
        if matched is None:
            lines.append({"center_y": center_y, "markers": [marker]})
        else:
            matched["markers"].append(marker)
            count = len(matched["markers"])
            matched["center_y"] = ((matched["center_y"] * (count - 1)) + center_y) / count

    compacted: list[dict] = []
    for line in lines:
        ordered = sorted(line["markers"], key=lambda marker: marker_bbox(marker)[0])
        segments: list[list[dict]] = []
        current_segment: list[dict] = []
        previous_right: int | None = None
        max_inline_gap = max(160, int(median_height * 8))
        for marker in ordered:
            bbox = marker_bbox(marker)
            if current_segment and previous_right is not None and bbox[0] - previous_right > max_inline_gap:
                segments.append(current_segment)
                current_segment = [marker]
            else:
                current_segment.append(marker)
            previous_right = max(previous_right or bbox[2], bbox[2])
        if current_segment:
            segments.append(current_segment)

        for segment in segments:
            ids = [marker.get("id") for marker in segment if marker.get("id") is not None]
            text = join_line_marker_text(segment)
            if text:
                compacted.append({"text": text, "ids": ids, "bbox": line_bbox(segment)})
    return sorted(compacted, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def is_probable_detail_noise_line(line: dict) -> bool:
    """내용이 없는 줄만 제거하고 의미상 중요도는 추출 모델에 맡긴다."""

    text = str(line.get("text") or "").strip()
    return not text


def append_limited_ocr_line(parts: list[str], index: int, line: dict, max_line_chars: int) -> None:
    text = str(line.get("text") or "").strip()
    if len(text) > max_line_chars:
        text = text[: max_line_chars - 1].rstrip() + "…"
    parts.append(f"{index}. {text}")


def detail_action_marker_candidates(
    markers: list[dict],
    limit: int,
    allowed_labels: list[str] | None = None,
) -> list[dict]:
    """본문 펼치기라고 명확히 선언된 마커만 결정론적 클릭 후보로 고른다."""

    labels = (
        [str(label) for label in allowed_labels if str(label).strip()]
        if allowed_labels is not None
        else []
    )
    normalized_labels = {re.sub(r"\s+", "", label).casefold() for label in labels}
    if not normalized_labels:
        return []
    primary: list[dict] = []
    seen: set[int] = set()
    for marker in sorted(markers, key=marker_prompt_rank):
        marker_id = marker.get("id")
        if marker_id in seen:
            continue
        text = str(marker.get("text") or "").strip()
        collapsed = re.sub(r"\s+", "", text).casefold()
        if collapsed in normalized_labels:
            primary.append(marker)
            seen.add(marker_id)
        if len(primary) >= limit:
            break
    return primary


def detail_reveal_controls(current_url: str) -> list[str]:
    """사이트가 명시한 본문 펼치기 컨트롤을 반환하며 빈 목록은 자동 클릭 금지를 뜻한다."""

    guidance = page_guidance_for_url(current_url, "job_detail")
    return [str(item) for item in guidance.get("reveal_controls", []) if str(item).strip()]


def draw_detail_lightweight_marker(draw: Any, marker: dict, color: tuple[int, int, int], font: Any) -> None:
    bbox = marker_bbox(marker)
    if bbox == [0, 0, 0, 0]:
        return
    x1, y1, x2, y2 = bbox
    pad = 4
    draw.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], outline=color, width=4)
    label = f"[{marker.get('id')}]"
    label_box = [x1 - pad, max(0, y1 - 30), x1 + 70, max(24, y1 - 4)]
    draw.rectangle(label_box, fill=color)
    draw.text((label_box[0] + 4, label_box[1] + 2), label, fill=(255, 255, 255), font=font)


def build_detail_lightweight_marked_image(
    image_path: Any,
    markers: list[dict],
    current_url: str,
    *,
    page_role: str = "",
) -> str:
    """상세 페이지 reasoning에는 클릭 후보만 표시한 가벼운 이미지를 만든다."""

    marker_texts = [marker.get("text") for marker in markers if isinstance(marker, dict)]
    if not current_url or not is_job_detail_context(
        current_url,
        page_role=page_role,
        marker_texts=marker_texts,
    ):
        return ""
    try:
        from pathlib import Path

        from PIL import Image, ImageDraw, ImageFont

        source_path = Path(image_path)
        if not source_path.exists():
            return ""
        action_limit = get_settings().vision.detail_action_marker_limit
        candidates = detail_action_marker_candidates(
            markers,
            action_limit,
            detail_reveal_controls(current_url),
        )
        if not candidates:
            return ""

        image = Image.open(source_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        for marker in candidates:
            draw_detail_lightweight_marker(draw, marker, (0, 120, 255), font)

        output_path = source_path.with_name(f"light_marked_{source_path.stem}.jpg")
        image.save(output_path, "JPEG", quality=88)
        logger.info(
            "Detail lightweight marked image prepared",
            markers_count=len(markers),
            highlighted_markers=len(candidates),
            output_path=str(output_path),
        )
        return str(output_path)
    except Exception as exc:
        logger.debug("detail lightweight marked image skipped", error=str(exc))
        return ""


def build_detail_section_context(markers: list[dict]) -> str:
    settings = get_settings().vision
    min_text_markers = settings.detail_section_min_text_markers
    max_lines = settings.detail_ocr_max_lines
    max_line_chars = settings.detail_section_max_line_chars

    text_marker_count = sum(
        1
        for marker in markers
        if str(marker.get("text") or "").strip() and not is_icon_marker(marker)
    )
    if text_marker_count < min_text_markers:
        return ""

    lines = [line for line in group_text_markers_into_lines(markers) if not is_probable_detail_noise_line(line)]
    if not lines:
        return ""

    parts = ["상세 페이지 OCR 본문(읽기용, 위에서 아래 순서. 원본 마커는 클릭/좌표용으로 유지됨):"]
    shown_lines = lines[:max_lines]
    for index, line in enumerate(shown_lines, start=1):
        append_limited_ocr_line(parts, index, line, max_line_chars)
    omitted_lines = max(0, len(lines) - len(shown_lines))
    if omitted_lines:
        parts.append(f"본문 압축으로 생략된 줄: {omitted_lines}개")

    parts.append(f"원본 텍스트 마커 {text_marker_count}개를 읽기용 줄 {len(lines)}개로 압축")
    return "\n".join(parts)


def detail_context_matches(
    context: dict[str, Any] | None,
    current_url: str,
    detail_key: str = "",
) -> bool:
    """카드 식별자가 있으면 URL 이동보다 공고 단위를 우선해 같은 상세 문맥인지 판단한다."""

    value = dict(context or {})
    stored_key = str(value.get("detail_key") or "").strip()
    resolved_key = str(detail_key or "").strip()
    if stored_key and resolved_key:
        return stored_key == resolved_key
    return bool(current_url and value.get("url") == current_url)


def detail_buffer_line_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def detail_lines_for_buffer(markers: list[dict]) -> list[dict]:
    return [
        line
        for line in group_text_markers_into_lines(markers)
        if not is_probable_detail_noise_line(line)
    ]


def new_job_detail_buffer(current_url: str, detail_key: str = "") -> dict[str, Any]:
    return {
        "url": current_url,
        "detail_key": detail_key,
        "lines": [],
        "seen_keys": [],
        "screens": [],
        "screen_evidence": [],
        "stats": {
            "screen_count": 0,
            "added_lines_last_screen": 0,
            "duplicate_lines_last_screen": 0,
            "total_lines": 0,
        },
    }


def update_job_detail_buffer(
    existing: dict[str, Any] | None,
    markers: list[dict],
    current_url: str,
    image_path: Any = "",
    *,
    page_role: str = "",
    detail_key: str = "",
) -> dict[str, Any]:
    """상세 페이지 OCR 본문 줄을 공고 단위 버퍼에 누적한다."""

    marker_texts = [marker.get("text") for marker in markers if isinstance(marker, dict)]
    if not current_url or not is_job_detail_context(
        current_url,
        page_role=page_role,
        marker_texts=marker_texts,
    ):
        return dict(existing or {})

    settings = get_settings().vision
    max_lines = settings.detail_buffer_max_lines
    max_line_chars = settings.detail_buffer_max_line_chars

    buffer = dict(existing or {})
    if not detail_context_matches(buffer, current_url, detail_key):
        buffer = new_job_detail_buffer(current_url, detail_key)
    lines = [dict(item) for item in (buffer.get("lines") or []) if isinstance(item, dict)]
    seen_keys = [str(item) for item in (buffer.get("seen_keys") or []) if str(item)]
    seen = set(seen_keys)
    added = 0
    duplicate = 0
    try:
        from pathlib import Path

        screen_name = Path(image_path).name if image_path else ""
        screen_path = str(Path(image_path).resolve()) if image_path else ""
    except Exception:
        screen_name = str(image_path or "")
        screen_path = screen_name

    for line in detail_lines_for_buffer(markers):
        text = str(line.get("text") or "").strip()
        if len(text) < 2:
            continue
        if len(text) > max_line_chars:
            text = text[: max_line_chars - 1].rstrip() + "…"
        key = detail_buffer_line_key(text)
        if not key:
            continue
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        seen_keys.append(key)
        lines.append(
            {
                "text": text,
                "bbox": line.get("bbox") or [0, 0, 0, 0],
                "first_screen": screen_name,
            }
        )
        added += 1
        if len(lines) >= max_lines:
            break

    screens = [str(item) for item in (buffer.get("screens") or []) if str(item)]
    if screen_path and (not screens or screens[-1] != screen_path):
        screens.append(screen_path)
    screen_evidence = [
        dict(item)
        for item in (buffer.get("screen_evidence") or [])
        if isinstance(item, dict)
    ]
    if screen_path and (
        not screen_evidence or screen_evidence[-1].get("path") != screen_path
    ):
        screen_evidence.append(
            {
                "path": screen_path,
                "added_lines": added,
                "duplicate_lines": duplicate,
            }
        )
    stats = dict(buffer.get("stats") or {})
    stats["screen_count"] = int(stats.get("screen_count") or 0) + 1
    stats["added_lines_last_screen"] = added
    stats["duplicate_lines_last_screen"] = duplicate
    stats["total_lines"] = len(lines)
    buffer.update(
        {
            "url": current_url,
            "detail_key": detail_key or str(buffer.get("detail_key") or ""),
            "lines": lines[:max_lines],
            "seen_keys": seen_keys[:max_lines],
            "screens": screens[-20:],
            "screen_evidence": screen_evidence[-20:],
            "stats": stats,
        }
    )
    logger.info(
        "Detail OCR buffer updated",
        url=current_url,
        added_lines=added,
        duplicate_lines=duplicate,
        total_lines=len(lines[:max_lines]),
        screen_count=stats["screen_count"],
    )
    return buffer


def compact_job_detail_buffer_context(
    state: "WorkerState",
    current_url: str,
    detail_key: str = "",
) -> str:
    if not current_url:
        return ""
    collection = state["collection"]
    observation = state["observation"]
    buffer = dict(collection.get("job_detail_buffer", {}) or {})
    if not detail_context_matches(buffer, current_url, detail_key):
        return ""
    if not buffer.get("lines") and not is_job_detail_context(
        current_url,
        page_role=str(observation.get("current_page_role") or ""),
    ):
        return ""
    stats = dict(buffer.get("stats") or {})
    lines = [item for item in (buffer.get("lines") or []) if isinstance(item, dict)]
    first_preview = [str(item.get("text") or "").strip() for item in lines[:8]]
    first_preview = [line for line in first_preview if line]
    preview = [str(item.get("text") or "").strip() for item in lines[-8:]]
    preview = [line for line in preview if line]
    followup = dict(collection.get("job_detail_followup") or {})
    followup_active = detail_context_matches(followup, current_url, detail_key)
    from agent.runtime.job_field_contract import (
        detail_coverage_status,
        field_contract_items,
        required_fields_from_state,
    )

    required_fields = required_fields_from_state(state)
    coverage = detail_coverage_status(
        dict(collection.get("job_detail_coverage") or {}),
        required_fields,
    )
    evidence_preview = {
        field: str(value)[:120]
        for field, value in coverage["field_evidence"].items()
        if field in coverage["found_fields"]
    }
    parts = [
        "상세 OCR 누적 상태:",
        f"- 누적 본문 줄 수: {len(lines)}",
        f"- 이번 화면 새 줄 수: {stats.get('added_lines_last_screen', 0)}",
        f"- 이번 화면 중복 줄 수: {stats.get('duplicate_lines_last_screen', 0)}",
        f"- 상세 화면 관찰 횟수: {stats.get('screen_count', 0)}",
        "- 필수 필드 계약: "
        + json.dumps(
            field_contract_items(required_fields),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "- 지금까지 확인한 필드 근거: "
        + json.dumps(
            evidence_preview,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "- 아직 확인하지 못한 필드: "
        + json.dumps(
            coverage["missing_fields"],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "- 상세 페이지에서는 중간 DB 추출을 위해 update_extracted_info를 호출하지 마십시오.",
        "- 더 읽어야 하면 scroll 또는 현재 사이트 안내에 선언된 상세 펼치기 버튼을 선택하고, "
        "현재 화면에서 확인한 필드를 observed_fields에 함께 넣으십시오.",
        "- 모든 필수 필드가 확인되었을 때만 finish_detail_reading을 호출하십시오. "
        "페이지 끝까지 확인해도 제공되지 않은 필드는 page_exhausted=true와 "
        "unavailable_fields로 명시하십시오.",
    ]
    if followup_active:
        parts.extend(
            [
                "- 직전 상세 완료는 필수 필드 근거 부족으로 거부되었습니다.",
                "- 거부된 누락 필드: "
                + json.dumps(
                    followup.get("missing_fields") or [],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "- 누적 OCR과 화면을 보고 원문 공고 이동 수단 또는 추가 본문 공개 수단이 있는지 판단하십시오.",
                "- 원문 이동 수단이 현재 화면에 없으면 위로 스크롤하고, 보이면 해당 마커를 click_marker로 선택하십시오.",
                "- URL을 추측해 open_browser를 호출하지 마십시오.",
            ]
        )
    if first_preview:
        parts.append(
            "- 처음 누적 본문 미리보기: "
            + json.dumps(first_preview, ensure_ascii=False, separators=(",", ":"))
        )
    if preview:
        parts.append(
            "- 최근 누적 본문 미리보기: "
            + json.dumps(preview, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n".join(parts) + "\n\n"


def detail_buffer_text(buffer: dict[str, Any]) -> str:
    """누적 OCR 줄을 최종 추출 모델의 입력 크기에 맞춰 렌더링한다."""

    max_chars = get_settings().vision.detail_final_ocr_max_chars
    lines = [item for item in (buffer.get("lines") or []) if isinstance(item, dict)]
    rendered: list[str] = []
    total = 0
    for index, line in enumerate(lines, start=1):
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        row = f"{index}. {text}"
        if total + len(row) + 1 > max_chars:
            break
        rendered.append(row)
        total += len(row) + 1
    return "\n".join(rendered)


__all__ = [
    "build_detail_lightweight_marked_image",
    "build_detail_section_context",
    "compact_job_detail_buffer_context",
    "detail_action_marker_candidates",
    "detail_buffer_text",
    "detail_buffer_line_key",
    "detail_context_matches",
    "detail_lines_for_buffer",
    "detail_reveal_controls",
    "draw_detail_lightweight_marker",
    "group_text_markers_into_lines",
    "is_icon_marker",
    "is_probable_detail_noise_line",
    "join_line_marker_text",
    "line_bbox",
    "marker_prompt_rank",
    "new_job_detail_buffer",
    "update_job_detail_buffer",
]
