"""작업자 공유 상태의 캡처·OCR 불변조건."""

from __future__ import annotations

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


__all__ = ["current_observation_matches_capture"]
