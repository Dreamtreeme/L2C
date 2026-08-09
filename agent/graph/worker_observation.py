"""작업자 그래프의 화면 캡처와 OCR 관찰 노드."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.runtime import Runtime

from agent.config import get_settings
from agent.observability.run_context import raise_if_cancelled
from agent.runtime.detail_runtime import (
    build_detail_lightweight_marked_image,
    build_detail_section_context,
    detail_context_matches,
    is_icon_marker,
    marker_prompt_rank,
    update_job_detail_buffer,
)
from agent.runtime.job_card_queue import job_detail_key_from_state
from agent.runtime.job_field_contract import (
    detail_coverage_matches,
    merge_job_detail_coverage,
)
from agent.runtime.site_context import is_job_detail_context
from agent.runtime.vision_worker_runtime import WorkerDependencies
from agent.runtime.worker_contracts import (
    WorkerState,
    WorkerStateUpdate,
    apply_worker_state_update,
)
from agent.runtime.worker_state import infer_current_page_role
from agent.utils.logger import logger
from agent.vision.screen_signature import (
    build_capture_context,
    compute_screen_phash_signature,
    compute_screen_signature,
)

_WAIT_ACTIONS = {
    "click_marker",
    "press_key",
    "open_browser",
    "go_back",
    "close_current_tab",
    "switch_tab",
}


def _build_ui_context(
    markers: list[dict[str, Any]],
    current_url: str,
    page_role: str,
) -> str:
    """상세 화면은 섹션을, 다른 화면은 제한된 마커 목록을 만든다."""

    marker_texts = [
        marker.get("text")
        for marker in markers
        if isinstance(marker, dict)
    ]
    if current_url and is_job_detail_context(
        current_url,
        page_role=page_role,
        marker_texts=marker_texts,
    ):
        section_context = build_detail_section_context(markers)
        if section_context:
            return section_context

    text_markers = sorted(
        [marker for marker in markers if not is_icon_marker(marker)],
        key=marker_prompt_rank,
    )
    icon_markers = sorted(
        [marker for marker in markers if is_icon_marker(marker)],
        key=marker_prompt_rank,
    )
    settings = get_settings().vision
    shown_text = text_markers[: settings.ui_text_marker_limit]
    shown_icons = icon_markers[: settings.ui_icon_marker_limit]
    parts: list[str] = []
    if shown_text:
        parts.append(
            "식별된 텍스트 요소:\n"
            + "\n".join(
                f"[id: {marker['id']}] {marker.get('text', '')}"
                for marker in shown_text
            )
        )
    if shown_icons:
        parts.append(
            "기타 아이콘/버튼 마커 ID 목록: "
            f"{[marker['id'] for marker in shown_icons]}"
        )
    omitted_text = len(text_markers) - len(shown_text)
    omitted_icons = len(icon_markers) - len(shown_icons)
    if omitted_text or omitted_icons:
        parts.append(
            "프롬프트 경량화를 위해 생략된 마커: "
            f"텍스트 {omitted_text}개, 아이콘 {omitted_icons}개"
        )
    return "\n".join(parts) if parts else "발견된 UI 마커 없음"


def _next_capture_identity(state: WorkerState) -> tuple[int, str]:
    """작업자 실행 안에서 읽기 쉬운 단조 증가 캡처 ID를 만든다."""

    try:
        sequence = max(
            0,
            int(state["observation"].get("capture_sequence", 0)),
        ) + 1
    except (TypeError, ValueError):
        sequence = 1
    run_id = str(state["request"].get("worker_run_id") or "").strip()
    prefix = f"{run_id}:" if run_id else ""
    return sequence, f"{prefix}capture:{sequence:04d}"


def _previous_screen_observation(state: WorkerState) -> dict[str, Any]:
    """새 캡처 전에 현재 OCR 관찰을 캡처 ID와 함께 보존한다."""

    observation = state["observation"]
    if (
        observation.get("ocr_complete")
        and observation.get("current_capture_id")
        and observation.get("current_screenshot")
        and observation.get("current_markers")
    ):
        return {
            "capture_id": str(observation.get("current_capture_id") or ""),
            "screenshot": str(observation.get("current_screenshot") or ""),
            "current_url": str(observation.get("current_url") or ""),
            "markers": list(observation.get("current_markers") or []),
            "ui_context": str(observation.get("ui_context") or ""),
            "marked_image": str(observation.get("marked_image") or ""),
            "screen_signature": dict(
                observation.get("screen_signature") or {}
            ),
            "page_role": str(observation.get("current_page_role") or ""),
            "analysis_mode": str(observation.get("analysis_mode") or "full"),
        }
    return dict(observation.get("previous_screen_observation") or {})


def capture_node(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> dict[str, Any]:
    """화면 변화 대기, 캡처, URL 읽기와 원본 pHash 계산만 수행한다."""

    raise_if_cancelled()
    perception = runtime.context.vision.get_perception()
    observation = state["observation"]
    transition_request = dict(
        state["transition"].get("transition_request", {}) or {}
    )
    pending_action = str(transition_request.get("action") or "")

    if pending_action in _WAIT_ACTIONS:
        before_screenshot = str(
            transition_request.get("before_screenshot") or ""
        )
        if before_screenshot:
            perception.wait_for_transition_change(before_screenshot)

    if pending_action == "type_in_marker":
        initial_wait = get_settings().browser.input_capture_initial_wait_sec
        image_path = perception.capture_usable_screen(
            initial_wait_sec=initial_wait
        )
    else:
        image_path = perception.capture_usable_screen()

    capture_quality = dict(perception.last_capture_quality)
    current_url = str(observation.get("current_url") or "")
    current_url_stale = bool(observation.get("current_url_stale", True))
    if (
        current_url_stale
        or not current_url
        or transition_request
    ):
        fetched_url = str(perception.get_current_url() or "")
        if fetched_url:
            current_url = fetched_url
            current_url_stale = False
        else:
            current_url_stale = True

    raw_signature = (
        compute_screen_phash_signature(image_path)
        if transition_request
        else {}
    )
    low_information = bool(capture_quality.get("low_information"))
    low_information_capture_count = (
        int(observation.get("low_information_capture_count") or 0) + 1
        if low_information
        else 0
    )
    capture_sequence, capture_id = _next_capture_identity(state)
    previous_observation = _previous_screen_observation(state)
    logger.info(
        "Worker screen captured",
        capture_id=capture_id,
        low_information=low_information,
        has_transition=bool(transition_request),
    )
    return {"observation": {
        "current_capture_id": capture_id,
        "ocr_capture_id": "",
        "capture_sequence": capture_sequence,
        "current_screenshot": str(image_path),
        "previous_screen_observation": previous_observation,
        "capture_quality": capture_quality,
        "raw_screen_signature": raw_signature,
        "analysis_mode": "",
        "ocr_complete": False,
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
    }}


def ocr_node(
    state: WorkerState,
    runtime: Runtime[WorkerDependencies],
) -> dict[str, Any]:
    """현재 캡처 한 장에 SoM/OCR을 한 번 실행하고 화면 문맥을 만든다."""

    raise_if_cancelled()
    observation_state = state["observation"]
    image_path_text = str(observation_state.get("current_screenshot") or "")
    if not image_path_text:
        raise RuntimeError("OCR을 실행할 캡처 이미지가 없습니다.")
    image_path = Path(image_path_text)

    analysis = runtime.context.vision.get_perception().analyze_ui(image_path)
    markers = list(analysis.get("markers", []) or [])
    marked_image = str(analysis.get("marked_image") or "")
    screen_signature = compute_screen_signature(image_path, markers)
    capture_context = build_capture_context(
        list(screen_signature.get("size") or []),
        int(analysis.get("content_top", 0) or 0),
    )
    if capture_context:
        screen_signature["capture_context"] = capture_context
    raw_signature = dict(
        observation_state.get("raw_screen_signature", {}) or {}
    )
    if raw_signature.get("phash"):
        screen_signature["phash"] = raw_signature["phash"]
        screen_signature["size"] = (
            raw_signature.get("size") or screen_signature.get("size")
        )

    current_url = str(observation_state.get("current_url") or "")
    page_role = infer_current_page_role(current_url, markers)
    ui_context = _build_ui_context(
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
    observation = {
        "marked_image": marked_image,
        "current_markers": markers,
        "ui_context": ui_context,
        "screen_signature": screen_signature,
        "current_page_role": page_role,
        "analysis_mode": analysis_mode,
        "ocr_complete": True,
        "ocr_capture_id": str(
            observation_state.get("current_capture_id") or ""
        ),
        "low_information_screen": False,
    }
    update: WorkerStateUpdate = {"observation": observation}
    updated_state = apply_worker_state_update(state, update)
    update.update(_collect_job_detail_observation(updated_state))
    return update


def _collect_job_detail_observation(
    state: WorkerState,
) -> WorkerStateUpdate:
    """현재 OCR 결과를 상세 공고 판독 상태에 한 번 반영한다."""

    observation = state["observation"]
    collection = state["collection"]
    if not observation.get("ocr_complete"):
        return {}
    current_url = str(observation.get("current_url") or "")
    detail_key = job_detail_key_from_state(state)
    detail_buffer = update_job_detail_buffer(
        dict(collection.get("job_detail_buffer", {}) or {}),
        list(observation.get("current_markers") or []),
        current_url,
        str(observation.get("current_screenshot") or ""),
        page_role=str(observation.get("current_page_role") or ""),
        detail_key=detail_key,
    )
    detail_followup = dict(collection.get("job_detail_followup", {}) or {})
    if detail_followup and not detail_context_matches(
        detail_followup,
        current_url,
        detail_key,
    ):
        detail_followup = {}
    return_to_results = dict(
        collection.get("return_to_job_results", {}) or {}
    )
    if return_to_results and return_to_results.get("url") != current_url:
        return_to_results = {}
    detail_coverage = dict(collection.get("job_detail_coverage", {}) or {})
    if (
        detail_context_matches(detail_buffer, current_url, detail_key)
        and not detail_coverage_matches(
            detail_coverage,
            current_url,
            detail_key,
        )
    ):
        detail_coverage = merge_job_detail_coverage(
            {},
            {},
            state=state,
            current_url=current_url,
            detail_key=detail_key,
        )
    return {
        "collection": {
            "job_detail_buffer": detail_buffer,
            "job_detail_coverage": detail_coverage,
            "job_detail_followup": detail_followup,
            "return_to_job_results": return_to_results,
        }
    }


__all__ = ["capture_node", "ocr_node"]
