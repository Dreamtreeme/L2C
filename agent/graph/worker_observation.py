"""작업자 그래프의 화면 캡처와 OCR 관찰 노드."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.config import get_settings

from agent.graph.state import GraphState
from agent.graph.worker_observation_context import build_ui_context
from agent.runtime.detail_runtime import build_detail_lightweight_marked_image
from agent.runtime.transition_runtime import raw_screen_phash_signature
from agent.utils.logger import logger


_WAIT_ACTIONS = {
    "click_marker",
    "press_key",
    "open_browser",
    "go_back",
    "close_current_tab",
    "switch_tab",
}


def _perception_engine() -> Any:
    from agent.graph.worker_resources import get_perception

    return get_perception()


def _next_capture_identity(state: GraphState) -> tuple[int, str]:
    """작업자 실행 안에서 읽기 쉬운 단조 증가 캡처 ID를 만든다."""

    try:
        sequence = max(0, int(state.get("capture_sequence", 0))) + 1
    except (TypeError, ValueError):
        sequence = 1
    run_id = str(state.get("worker_run_id") or "").strip()
    try:
        attempt_index = max(0, int(state.get("worker_attempt_index", 0)))
    except (TypeError, ValueError):
        attempt_index = 0
    prefix = f"{run_id}:attempt:{attempt_index:02d}:" if run_id else ""
    return sequence, f"{prefix}capture:{sequence:04d}"


def _previous_screen_observation(state: GraphState) -> dict[str, Any]:
    """새 캡처 전에 현재 OCR 관찰을 캡처 ID와 함께 보존한다."""

    if (
        state.get("ocr_complete")
        and state.get("current_capture_id")
        and state.get("current_screenshot")
        and state.get("current_markers")
    ):
        return {
            "capture_id": str(state.get("current_capture_id") or ""),
            "screenshot": str(state.get("current_screenshot") or ""),
            "current_url": str(state.get("current_url") or ""),
            "markers": list(state.get("current_markers") or []),
            "ui_context": str(state.get("ui_context") or ""),
            "marked_image": str(state.get("marked_image") or ""),
            "screen_signature": dict(state.get("screen_signature") or {}),
            "page_role": str(state.get("current_page_role") or ""),
            "analysis_mode": str(state.get("analysis_mode") or "full"),
        }
    return dict(state.get("previous_screen_observation") or {})


def capture_screen_node(state: GraphState) -> dict[str, Any]:
    """화면 변화 대기, 캡처, URL 읽기와 원본 pHash 계산만 수행한다."""

    from agent.application.run_context import raise_if_cancelled

    raise_if_cancelled()
    perception = _perception_engine()
    pending = dict(state.get("pending_transition", {}) or {})
    pending_action = str(pending.get("action") or "")

    if pending_action in _WAIT_ACTIONS:
        wait_for_change = getattr(perception, "wait_for_transition_change", None)
        before_screenshot = str(pending.get("before_screenshot") or "")
        if callable(wait_for_change) and before_screenshot:
            try:
                wait_for_change(before_screenshot)
            except Exception as exc:
                logger.debug("Transition screen change wait skipped", error=str(exc))

    capture_usable = getattr(perception, "capture_usable_screen", None)
    if callable(capture_usable):
        if pending_action == "type_in_marker":
            try:
                initial_wait = max(
                    0.0,
                    get_settings().browser.input_capture_initial_wait_sec,
                )
            except ValueError:
                initial_wait = 0.7
            image_path = capture_usable(initial_wait_sec=initial_wait)
        else:
            image_path = capture_usable()
    else:
        image_path = perception.capture_screen()

    capture_quality = dict(getattr(perception, "last_capture_quality", {}) or {})
    current_url = str(state.get("current_url") or "")
    current_url_stale = bool(state.get("current_url_stale", True))
    read_current_url = getattr(perception, "get_current_url", None)
    if (current_url_stale or not current_url or pending) and callable(read_current_url):
        fetched_url = str(read_current_url() or "")
        if fetched_url:
            current_url = fetched_url
            current_url_stale = False
        else:
            current_url_stale = True

    raw_signature = raw_screen_phash_signature(image_path) if pending else {}
    low_information = bool(capture_quality.get("low_information"))
    low_information_capture_count = (
        int(state.get("low_information_capture_count") or 0) + 1
        if low_information
        else 0
    )
    capture_sequence, capture_id = _next_capture_identity(state)
    previous_observation = _previous_screen_observation(state)
    logger.info(
        "Worker screen captured",
        capture_id=capture_id,
        low_information=low_information,
        has_transition=bool(pending),
    )
    return {
        "current_capture_id": capture_id,
        "capture_sequence": capture_sequence,
        "current_screenshot": str(image_path),
        "previous_screen_observation": previous_observation,
        "capture_quality": capture_quality,
        "raw_screen_signature": raw_signature,
        "analysis_mode": "",
        "ocr_complete": False,
        "recent_images": [str(image_path)],
        "current_url": current_url,
        "current_url_stale": current_url_stale,
        "low_information_screen": low_information,
        "low_information_capture_count": low_information_capture_count,
        # 새 캡처의 OCR이 끝나기 전에는 직전 화면 인식을 재사용하지 않는다.
        "ui_context": "",
        "current_markers": [],
        "marked_image": "",
        "screen_signature": {},
        "current_page_role": "",
    }


def analyze_screen_node(state: GraphState) -> dict[str, Any]:
    """현재 캡처 한 장에 SoM/OCR을 한 번 실행하고 화면 문맥을 만든다."""

    from agent.application.run_context import raise_if_cancelled
    from agent.graph.worker_state import infer_current_page_role
    from agent.vision.screen_signature import build_capture_context, compute_screen_signature

    raise_if_cancelled()
    image_path_text = str(state.get("current_screenshot") or "")
    if not image_path_text:
        raise RuntimeError("OCR을 실행할 캡처 이미지가 없습니다.")
    image_path = Path(image_path_text)

    analysis = _perception_engine().analyze_ui(image_path)
    markers = list(analysis.get("markers", []) or [])
    marked_image = str(analysis.get("marked_image") or "")
    screen_signature: dict[str, Any] = {}
    try:
        screen_signature = compute_screen_signature(image_path, markers)
        capture_context = build_capture_context(
            list(screen_signature.get("size") or []),
            int(analysis.get("content_top", 0) or 0),
        )
        if capture_context:
            screen_signature["capture_context"] = capture_context
        raw_signature = dict(state.get("raw_screen_signature", {}) or {})
        if raw_signature.get("phash"):
            screen_signature["phash"] = raw_signature["phash"]
            screen_signature["size"] = (
                raw_signature.get("size") or screen_signature.get("size")
            )
    except Exception as exc:
        logger.debug("Screen signature skipped", error=str(exc))

    current_url = str(state.get("current_url") or "")
    page_role = infer_current_page_role(current_url, markers)
    ui_context = build_ui_context(
        markers,
        current_url=current_url,
        page_role=page_role,
    )
    lightweight_image = build_detail_lightweight_marked_image(
        image_path,
        markers,
        current_url,
        page_role=page_role,
    )
    if lightweight_image:
        marked_image = str(lightweight_image)

    analysis_mode = str(analysis.get("analysis_mode") or "full")
    logger.info(
        "Worker screen OCR completed",
        marker_count=len(markers),
        analysis_mode=analysis_mode,
    )
    return {
        "marked_image": marked_image,
        "current_markers": markers,
        "ui_context": ui_context,
        "screen_signature": screen_signature,
        "current_page_role": page_role,
        "analysis_mode": analysis_mode,
        "ocr_complete": True,
        "low_information_screen": False,
    }


def observe_screen_cycle(state: GraphState) -> dict[str, Any]:
    """초기 화면 준비와 노드 단위 테스트에서 한 관찰 사이클을 합성한다."""

    from agent.graph.worker_collection import apply_observation_node
    from agent.graph.worker_selection import select_deterministic_action_node
    from agent.graph.worker_transition import evaluate_transition_node

    working = dict(state)
    updates: dict[str, Any] = {}

    def apply(result: dict[str, Any]) -> None:
        working.update(result)
        updates.update(result)

    apply(capture_screen_node(working))
    apply(evaluate_transition_node(working))
    apply(select_deterministic_action_node(working))
    if working.get("pending_action") is not None or working.get("low_information_screen"):
        return updates
    if not working.get("ocr_required"):
        return updates

    apply(analyze_screen_node(working))
    apply(evaluate_transition_node(working))
    apply(apply_observation_node(working))
    apply(select_deterministic_action_node(working))
    return updates


__all__ = ["analyze_screen_node", "capture_screen_node", "observe_screen_cycle"]
