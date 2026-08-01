"""SoM/OCR 마커를 화면 판단용 텍스트 문맥으로 변환한다."""

from __future__ import annotations

from typing import Any

from agent.config import get_settings
from agent.runtime.detail_runtime import (
    build_detail_section_context,
    is_icon_marker,
    marker_prompt_rank,
)
from agent.runtime.site_context import is_job_detail_context


def build_ui_context(
    markers: list[dict[str, Any]],
    current_url: str = "",
    page_role: str = "",
) -> str:
    """상세 페이지는 섹션 문맥을, 그 외 화면은 제한된 마커 목록을 만든다."""

    marker_texts = [marker.get("text") for marker in markers if isinstance(marker, dict)]
    settings = get_settings().vision
    if (
        current_url
        and is_job_detail_context(
            current_url,
            page_role=page_role,
            marker_texts=marker_texts,
        )
    ):
        section_context = build_detail_section_context(markers)
        if section_context:
            return section_context

    text_markers = []
    icon_markers = []
    for marker in markers:
        if is_icon_marker(marker):
            icon_markers.append(marker)
        else:
            text_markers.append(marker)

    text_markers = sorted(text_markers, key=marker_prompt_rank)
    icon_markers = sorted(icon_markers, key=marker_prompt_rank)
    shown_text_markers = text_markers[: settings.ui_text_marker_limit]
    shown_icon_markers = icon_markers[: settings.ui_icon_marker_limit]

    parts = []
    if shown_text_markers:
        parts.append(
            "식별된 텍스트 요소:\n"
            + "\n".join(
                f"[id: {marker['id']}] {marker.get('text', '')}"
                for marker in shown_text_markers
            )
        )
    if shown_icon_markers:
        parts.append(
            f"기타 아이콘/버튼 마커 ID 목록: {[marker['id'] for marker in shown_icon_markers]}"
        )
    omitted_text = max(0, len(text_markers) - len(shown_text_markers))
    omitted_icon = max(0, len(icon_markers) - len(shown_icon_markers))
    if omitted_text or omitted_icon:
        parts.append(
            "프롬프트 경량화를 위해 생략된 마커: "
            f"텍스트 {omitted_text}개, 아이콘 {omitted_icon}개"
        )
    return "\n".join(parts) if parts else "발견된 UI 마커 없음"


__all__ = ["build_ui_context"]
