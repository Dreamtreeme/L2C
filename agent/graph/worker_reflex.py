"""활성 레시피의 ROI 검증과 재생을 작업자 그래프에 연결한다."""

from __future__ import annotations

from typing import Any

from agent.graph.state import GraphState
from agent.runtime.reflex_runtime import attempt_reflex_replay


def reflex_node(state: GraphState) -> dict[str, Any]:
    """현재 화면에서 안전하게 재생 가능한 원자 레시피를 찾는다."""

    return attempt_reflex_replay(state)


__all__ = ["reflex_node"]
