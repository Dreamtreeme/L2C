"""작업자 공유 상태의 캡처·OCR 불변조건."""

from __future__ import annotations

from typing import Any

from agent.graph.state import GraphState


def current_observation_matches_capture(state: GraphState) -> bool:
    """OCR 관찰이 현재 캡처에서 생성되거나 검증되어 재사용됐는지 확인한다."""

    if not state.get("ocr_complete"):
        return False
    current_capture_id = str(state.get("current_capture_id") or "")
    ocr_capture_id = str(state.get("ocr_capture_id") or "")
    if current_capture_id or ocr_capture_id:
        return bool(
            current_capture_id
            and ocr_capture_id
            and current_capture_id == ocr_capture_id
        )
    # 캡처 장치를 사용하지 않는 순수 정책 입력에는 식별자가 없다.
    return True


def current_observation_errors(state: GraphState) -> list[str]:
    """디버깅과 계약 테스트에 사용할 현재 상태 위반 목록을 반환한다."""

    errors: list[str] = []
    current_capture_id = str(state.get("current_capture_id") or "")
    ocr_capture_id = str(state.get("ocr_capture_id") or "")
    if state.get("ocr_complete") and not current_observation_matches_capture(state):
        errors.append("ocr_capture_mismatch")
    if not state.get("ocr_complete") and ocr_capture_id:
        errors.append("ocr_capture_without_completed_ocr")
    if state.get("active_reflex_recipe"):
        active_recipe = dict(state.get("active_reflex_recipe") or {})
        current_index = int(active_recipe.get("current_transition_index") or 0)
        transition_count = int(active_recipe.get("transition_count") or 0)
        if transition_count <= 0 or current_index < 0 or current_index >= transition_count:
            errors.append("active_reflex_transition_out_of_range")
    if current_capture_id and not state.get("current_screenshot"):
        errors.append("capture_without_screenshot")
    return errors


def observation_capture_update(state: GraphState) -> dict[str, Any]:
    """현재 캡처를 OCR 관찰의 소유 캡처로 확정한다."""

    capture_id = str(state.get("current_capture_id") or "")
    if not capture_id:
        return {}
    return {"ocr_capture_id": capture_id}


__all__ = [
    "current_observation_errors",
    "current_observation_matches_capture",
    "observation_capture_update",
]
