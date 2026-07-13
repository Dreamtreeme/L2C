"""Reflex 학습 저장소 payload 정리 유틸."""

from __future__ import annotations

from typing import Any


STATE_DEBUG_FIELDS = {
    "state_key",
    "reflex_state_key",
    "state_anchors",
    "before_state_key",
    "page_state_key",
    "screen_signature",
}


def strip_state_debug_fields(value: Any) -> Any:
    """저장용 payload에서 화면상태 키와 전체 화면 디버그 서명을 제거한다."""

    if isinstance(value, dict):
        return {
            key: strip_state_debug_fields(child)
            for key, child in value.items()
            if key not in STATE_DEBUG_FIELDS
        }
    if isinstance(value, list):
        return [strip_state_debug_fields(item) for item in value]
    return value
