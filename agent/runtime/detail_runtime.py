"""상세 페이지 OCR을 읽기용 줄과 누적 버퍼로 변환한다."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import AIMessage

from agent.graph.action_request import build_action_message
from agent.runtime.site_context import looks_like_job_detail_url
from agent.utils.logger import logger
from agent.vision.marker_geometry import marker_bbox


def env_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def marker_prompt_rank(marker: dict) -> tuple[int, int, int]:
    text = marker.get("text", "")
    bbox = marker.get("bbox", [0, 0, 0, 0])
    y = int(bbox[1]) if len(bbox) == 4 else 0
    x = int(bbox[0]) if len(bbox) == 4 else 0
    lowered = text.lower()
    important_terms = (
        "검색", "채용", "포지션", "데이터", "엔지니어", "개발", "로그인",
        "닫기", "x", "원티드", "wanted", "지원", "상세", "회사",
    )
    priority = 0 if any(term in lowered for term in important_terms) else 1
    return (priority, y, x)


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
    """브라우저·사이트 헤더와 상세 동작 버튼처럼 본문이 아닌 줄을 거른다."""

    bbox = line.get("bbox") or [0, 0, 0, 0]
    text = str(line.get("text") or "").strip()
    if not text:
        return True
    collapsed = re.sub(r"\s+", "", text)
    lowered = collapsed.lower()
    if len(bbox) == 4 and bbox[1] < 120:
        return True
    browser_terms = ("youtube", "github", "gmail", "naver", "chzzk", "모든북마크")
    if sum(1 for term in browser_terms if term in lowered) >= 2:
        return True
    site_nav_terms = ("wanted", "채용", "이력서", "교육이벤트", "콘텐츠", "소셜", "프리랜서", "회원가입", "기업서비스")
    if ("wanted" in lowered or "원티드" in collapsed) and sum(1 for term in site_nav_terms if term in lowered) >= 2:
        return True
    if any(term in collapsed for term in ("상세정보더보기", "지원하기", "합격확률확인하기", "북마크", "공유하기")):
        return True
    if any(term in collapsed for term in ("회원가입/로그인", "회원가입로그인", "기업서비스", "합격확률", "이포지션나의합격확률은")):
        return True
    if len(bbox) == 4 and bbox[1] < 180 and any(
        term in collapsed
        for term in ("wanted", "원티드", "채용", "이력서", "교육이벤트", "콘텐츠", "소셜", "프리랜서", "회원가입", "기업서비스")
    ):
        return True
    return text in {"wanted", "원티드", "채용", "이력서", "교육·이벤트", "콘텐츠", "소셜", "프리랜서", "더보기"}


def append_limited_ocr_line(parts: list[str], index: int, line: dict, max_line_chars: int) -> None:
    text = str(line.get("text") or "").strip()
    if len(text) > max_line_chars:
        text = text[: max_line_chars - 1].rstrip() + "…"
    parts.append(f"{index}. {text}")


def detail_action_marker_candidates(markers: list[dict], limit: int) -> list[dict]:
    primary_terms = ("상세 정보 더 보기", "더 보기", "더보기", "상세정보더보기")
    primary: list[dict] = []
    seen: set[int] = set()
    for marker in sorted(markers, key=marker_prompt_rank):
        marker_id = marker.get("id")
        if marker_id in seen:
            continue
        text = str(marker.get("text") or "").strip()
        collapsed = re.sub(r"\s+", "", text)
        bbox = marker_bbox(marker)
        if any(term.replace(" ", "") in collapsed for term in primary_terms):
            if "상세" in collapsed or (len(bbox) == 4 and bbox[1] > 240):
                primary.append(marker)
                seen.add(marker_id)
        if len(primary) >= limit:
            break
    return primary


def detail_lightweight_marked_image_enabled() -> bool:
    return env_enabled("VISION_DETAIL_LIGHTWEIGHT_MARKED_IMAGE_ENABLED", True)


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


def build_detail_lightweight_marked_image(image_path: Any, markers: list[dict], current_url: str) -> str:
    """상세 페이지 reasoning에는 클릭 후보만 표시한 가벼운 이미지를 만든다."""

    if not current_url or not looks_like_job_detail_url(current_url):
        return ""
    if not detail_lightweight_marked_image_enabled():
        return ""
    try:
        from pathlib import Path

        from PIL import Image, ImageDraw, ImageFont

        source_path = Path(image_path)
        if not source_path.exists():
            return ""
        try:
            action_limit = int(os.getenv("VISION_DETAIL_ACTION_MARKER_LIMIT", "35"))
        except ValueError:
            action_limit = 35
        candidates = detail_action_marker_candidates(markers, action_limit)

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
    try:
        min_text_markers = int(os.getenv("VISION_DETAIL_SECTION_MIN_TEXT_MARKERS", "120"))
        max_lines = int(os.getenv("VISION_DETAIL_OCR_MAX_LINES", "90"))
        max_line_chars = int(os.getenv("VISION_DETAIL_SECTION_MAX_LINE_CHARS", "180"))
        action_limit = int(os.getenv("VISION_DETAIL_ACTION_MARKER_LIMIT", "35"))
    except ValueError:
        min_text_markers = 120
        max_lines = 90
        max_line_chars = 180
        action_limit = 35

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

    action_markers = detail_action_marker_candidates(markers, action_limit)
    if action_markers:
        parts.append("수집 진행용 클릭 후보:")
        for marker in action_markers:
            parts.append(f"[id: {marker.get('id')}] {marker.get('text', '')}")

    parts.append(f"원본 텍스트 마커 {text_marker_count}개를 읽기용 줄 {len(lines)}개로 압축")
    return "\n".join(parts)


def detail_ocr_buffer_enabled() -> bool:
    return env_enabled("VISION_DETAIL_OCR_BUFFER_ENABLED", True)


def detail_buffer_line_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def detail_lines_for_buffer(markers: list[dict]) -> list[dict]:
    return [
        line
        for line in group_text_markers_into_lines(markers)
        if not is_probable_detail_noise_line(line)
    ]


def new_detail_ocr_buffer(current_url: str) -> dict[str, Any]:
    return {
        "url": current_url,
        "lines": [],
        "seen_keys": [],
        "screens": [],
        "stats": {
            "screen_count": 0,
            "added_lines_last_screen": 0,
            "duplicate_lines_last_screen": 0,
            "total_lines": 0,
        },
    }


def update_detail_ocr_buffer(
    existing: dict[str, Any] | None,
    markers: list[dict],
    current_url: str,
    image_path: Any = "",
) -> dict[str, Any]:
    """상세 페이지 OCR 본문 줄을 URL 단위 버퍼에 누적한다."""

    if not detail_ocr_buffer_enabled():
        return dict(existing or {})
    if not current_url or not looks_like_job_detail_url(current_url):
        return dict(existing or {})

    try:
        max_lines = int(os.getenv("VISION_DETAIL_OCR_BUFFER_MAX_LINES", "260"))
        max_line_chars = int(os.getenv("VISION_DETAIL_OCR_BUFFER_MAX_LINE_CHARS", "220"))
    except ValueError:
        max_lines = 260
        max_line_chars = 220

    buffer = dict(existing or {})
    if buffer.get("url") != current_url:
        buffer = new_detail_ocr_buffer(current_url)
    lines = [dict(item) for item in (buffer.get("lines") or []) if isinstance(item, dict)]
    seen_keys = [str(item) for item in (buffer.get("seen_keys") or []) if str(item)]
    seen = set(seen_keys)
    added = 0
    duplicate = 0
    try:
        from pathlib import Path

        screen_name = Path(image_path).name if image_path else ""
    except Exception:
        screen_name = str(image_path or "")

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
    if screen_name and (not screens or screens[-1] != screen_name):
        screens.append(screen_name)
    stats = dict(buffer.get("stats") or {})
    stats["screen_count"] = int(stats.get("screen_count") or 0) + 1
    stats["added_lines_last_screen"] = added
    stats["duplicate_lines_last_screen"] = duplicate
    stats["total_lines"] = len(lines)
    buffer.update(
        {
            "url": current_url,
            "lines": lines[:max_lines],
            "seen_keys": seen_keys[:max_lines],
            "screens": screens[-20:],
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


def detail_page_policy_enabled() -> bool:
    return env_enabled("VISION_DETAIL_PAGE_POLICY_ENABLED", True)


def detail_policy_limits() -> tuple[int, int, int, int]:
    try:
        min_screens = int(os.getenv("VISION_DETAIL_POLICY_MIN_SCREENS", "3"))
        max_screens = int(os.getenv("VISION_DETAIL_POLICY_MAX_SCREENS", "4"))
        min_added_lines = int(os.getenv("VISION_DETAIL_POLICY_MIN_ADDED_LINES", "8"))
        reveal_min_screen = int(os.getenv("VISION_DETAIL_POLICY_REVEAL_MIN_SCREEN", "2"))
    except ValueError:
        min_screens = 3
        max_screens = 4
        min_added_lines = 8
        reveal_min_screen = 2
    return min_screens, max(max_screens, min_screens), min_added_lines, reveal_min_screen


def detail_page_policy_message(
    current_url: str,
    markers: list[dict],
    detail_ocr_buffer: dict[str, Any],
    *,
    transition_status: str = "",
) -> tuple[AIMessage | None, dict[str, Any]]:
    """상세 페이지의 반복 읽기 행동만 결정론적으로 선택한다."""

    if not detail_page_policy_enabled() or transition_status == "pending":
        return None, {}
    if not current_url or not looks_like_job_detail_url(current_url):
        return None, {}
    if detail_ocr_buffer.get("url") != current_url:
        return None, {}

    stats = dict(detail_ocr_buffer.get("stats") or {})
    screen_count = int(stats.get("screen_count") or 0)
    added_lines = int(stats.get("added_lines_last_screen") or 0)
    duplicate_lines = int(stats.get("duplicate_lines_last_screen") or 0)
    total_lines = len(
        [item for item in (detail_ocr_buffer.get("lines") or []) if isinstance(item, dict)]
    )
    min_screens, max_screens, min_added_lines, reveal_min_screen = detail_policy_limits()

    candidates = detail_action_marker_candidates(markers, 1)
    if candidates and screen_count >= reveal_min_screen:
        marker = candidates[0]
        marker_id = marker.get("id")
        if marker_id is not None:
            trace = {
                "policy": "detail_reveal",
                "marker_id": marker_id,
                "screen_count": screen_count,
                "added_lines_last_screen": added_lines,
                "duplicate_lines_last_screen": duplicate_lines,
                "total_lines": total_lines,
            }
            message = build_action_message(
                "page_policy",
                "detail reveal",
                [
                    {
                        "name": "click_marker",
                        "args": {
                            "marker_id": marker_id,
                            "page_role": "job_detail",
                            "target_role": "button",
                            "target_component": "expand_detail_button",
                            "target_label": str(marker.get("text") or "상세 정보 더 보기"),
                            "reason": "상세 페이지 본문을 더 펼치기 위해 보이는 상세 정보 버튼을 클릭합니다.",
                            "expected_after": "상세 페이지 본문이 펼쳐지거나 추가 본문이 노출됩니다.",
                            "risk_level": "safe_read",
                            "needs_user_confirmation": False,
                            "_transition_source": "page_policy",
                        },
                        "id": "detail_policy_reveal",
                    }
                ],
            )
            return message, trace

    should_scroll = screen_count < max_screens and (
        screen_count < min_screens or added_lines >= min_added_lines
    )
    if should_scroll:
        trace = {
            "policy": "detail_scroll",
            "screen_count": screen_count,
            "added_lines_last_screen": added_lines,
            "duplicate_lines_last_screen": duplicate_lines,
            "total_lines": total_lines,
        }
        message = build_action_message(
            "page_policy",
            "detail scroll",
            [
                {
                    "name": "scroll",
                    "args": {
                        "direction": "down",
                        "page_role": "job_detail",
                        "reason": "상세 페이지 OCR 본문을 더 누적하기 위해 다음 화면으로 이동합니다.",
                        "expected_after": "상세 페이지의 아래쪽 본문이 화면에 나타납니다.",
                        "risk_level": "safe_read",
                        "needs_user_confirmation": False,
                        "_transition_source": "page_policy",
                    },
                    "id": "detail_policy_scroll",
                }
            ],
        )
        return message, trace

    trace = {
        "policy": "detail_finish",
        "screen_count": screen_count,
        "added_lines_last_screen": added_lines,
        "duplicate_lines_last_screen": duplicate_lines,
        "total_lines": total_lines,
        "max_screens": max_screens,
    }
    message = build_action_message(
        "page_policy",
        "detail finish",
        [
            {
                "name": "finish_detail_reading",
                "args": {
                    "page_role": "job_detail",
                    "detail_complete": True,
                    "reason": "상세 페이지 OCR 본문을 충분히 누적했으므로 읽기를 종료합니다.",
                },
                "id": "detail_policy_finish",
            }
        ],
    )
    return message, trace


def compact_detail_ocr_buffer_context(state: dict, current_url: str) -> str:
    if not detail_ocr_buffer_enabled() or not current_url or not looks_like_job_detail_url(current_url):
        return ""
    buffer = dict(state.get("detail_ocr_buffer", {}) or {})
    if buffer.get("url") != current_url:
        return ""
    stats = dict(buffer.get("stats") or {})
    lines = [item for item in (buffer.get("lines") or []) if isinstance(item, dict)]
    preview = [str(item.get("text") or "").strip() for item in lines[-8:]]
    preview = [line for line in preview if line]
    parts = [
        "상세 OCR 누적 상태:",
        f"- 누적 본문 줄 수: {len(lines)}",
        f"- 이번 화면 새 줄 수: {stats.get('added_lines_last_screen', 0)}",
        f"- 이번 화면 중복 줄 수: {stats.get('duplicate_lines_last_screen', 0)}",
        f"- 상세 화면 관찰 횟수: {stats.get('screen_count', 0)}",
        "- 상세 페이지에서는 중간 DB 추출을 위해 update_extracted_info를 호출하지 마십시오.",
        "- 더 읽어야 하면 scroll 또는 보이는 상세 펼치기 버튼 클릭을 선택하십시오.",
        "- 현재 공고 정보가 충분하면 finish_detail_reading(page_role=\"job_detail\", detail_complete=true)을 호출하십시오.",
    ]
    if preview:
        parts.append(
            "- 최근 누적 본문 미리보기: "
            + json.dumps(preview, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n".join(parts) + "\n\n"


def detail_buffer_text(buffer: dict[str, Any]) -> str:
    """누적 OCR 줄을 최종 추출 모델의 입력 크기에 맞춰 렌더링한다."""

    try:
        max_chars = int(os.getenv("VISION_DETAIL_FINAL_OCR_MAX_CHARS", "16000"))
    except ValueError:
        max_chars = 16000
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
    "compact_detail_ocr_buffer_context",
    "detail_action_marker_candidates",
    "detail_buffer_text",
    "detail_buffer_line_key",
    "detail_lightweight_marked_image_enabled",
    "detail_lines_for_buffer",
    "detail_ocr_buffer_enabled",
    "detail_page_policy_enabled",
    "detail_page_policy_message",
    "detail_policy_limits",
    "draw_detail_lightweight_marker",
    "env_enabled",
    "group_text_markers_into_lines",
    "is_icon_marker",
    "is_probable_detail_noise_line",
    "join_line_marker_text",
    "line_bbox",
    "marker_prompt_rank",
    "new_detail_ocr_buffer",
    "update_detail_ocr_buffer",
]
