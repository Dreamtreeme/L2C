"""E2E 추적과 대시보드가 공유하는 실행 단계 이름."""

from __future__ import annotations


COMPONENT_STAGE = {
    "ocr_startup": "perception",
    "ocr_request": "perception",
    "worker_prepare_screen": "browser_prepare",
    "worker_graph": "collection",
    "vision_worker": "collection",
    "worker_review": "review",
    "job_persistence": "persistence",
    "browser_cleanup": "cleanup",
    "graph:capture": "perception",
    "graph:ocr": "perception",
    "graph:transition": "transition",
    "graph:collection": "collection",
    "graph:selection": "selection",
    "graph:recording": "recording",
    "graph:reasoning": "reasoning",
    "graph:reflex": "reflex",
    "graph:execution": "execution",
}


def stage_for_component(component: str) -> str:
    name = str(component or "unknown")
    if name in COMPONENT_STAGE:
        return COMPONENT_STAGE[name]
    if name.startswith("graph:execution ("):
        return "execution"
    return "other"


__all__ = ["COMPONENT_STAGE", "stage_for_component"]
