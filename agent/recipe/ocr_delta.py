"""OCR 마커 변화량 계산."""

from __future__ import annotations

from agent.recipe.state_key import canonical_anchor_text


def marker_text_set(markers: list[dict]) -> set[str]:
    """OCR 마커 목록을 상태 비교용 canonical 텍스트 집합으로 축약한다."""
    texts: set[str] = set()
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        text = canonical_anchor_text(marker.get("text"))
        if len(text) < 2:
            continue
        if text.startswith("상호작용 가능한 요소"):
            continue
        texts.add(text)
    return texts


def diff_marker_texts(previous_texts, current_markers: list[dict]) -> dict[str, list[str]]:
    """이전 OCR 텍스트 집합과 현재 마커를 비교해 추가/삭제된 텍스트만 반환한다."""
    previous = set(previous_texts or [])
    current = marker_text_set(current_markers)
    return {
        "current": sorted(current),
        "added": sorted(current - previous),
        "removed": sorted(previous - current),
    }
