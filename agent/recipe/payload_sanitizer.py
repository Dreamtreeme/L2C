"""Reflex 학습 저장소 payload 정리 유틸."""

from __future__ import annotations

from typing import Any


FULL_SCREEN_SIGNATURE_FIELDS = {
    "screen_signature",
}

REPLAY_RUNTIME_FIELDS = {
    "worker_run_id",
    "capture_sequence",
    "current_capture_id",
    "decision_capture_id",
    "from_capture_id",
    "to_capture_id",
}


def strip_full_screen_signatures(value: Any) -> Any:
    """저장용 payload에서 ROI 재생에 필요 없는 전체 화면 서명을 제거한다."""

    if isinstance(value, dict):
        return {
            key: strip_full_screen_signatures(child)
            for key, child in value.items()
            if key not in FULL_SCREEN_SIGNATURE_FIELDS
        }
    if isinstance(value, list):
        return [strip_full_screen_signatures(item) for item in value]
    return value


def strip_replay_runtime_fields(value: Any) -> Any:
    """활성 레시피에서 특정 실행에만 유효한 추적 식별자를 제거한다."""

    if isinstance(value, dict):
        return {
            key: strip_replay_runtime_fields(child)
            for key, child in value.items()
            if key not in REPLAY_RUNTIME_FIELDS
        }
    if isinstance(value, list):
        return [strip_replay_runtime_fields(item) for item in value]
    return value
