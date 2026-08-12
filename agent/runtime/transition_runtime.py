"""행동 전후 프레임 변화와 저장 상태 검증에 필요한 관찰 기록을 만든다."""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import parse_qsl, urlparse

from agent.config import get_settings
from agent.runtime.worker_contracts import (
    CompletedTransitionObservation,
    ObservationState,
    TransitionRequest,
    TransitionResult,
    WorkerState,
    action_event_results,
)
from agent.utils.logger import logger
from agent.utils.text import normalize_text, url_template
from agent.vision.screen_signature import (
    compact_screen_context_signature,
    compute_screen_phash_signature,
    hamming_distance,
)
from shared.schema.feedback_schema import ExecutionEvent
from shared.schema.recipe_schema import ExperienceTransition


def detect_two_screen_transition_cycle(
    observations: list[ExperienceTransition],
) -> dict[str, Any]:
    """최근 전환 화면이 A-B-A-B로 반복됐는지 pHash로 확인한다."""

    recent = [
        transition
        for transition in observations
        if transition.evidence and transition.evidence.screenshot
    ][-4:]
    if len(recent) < 4:
        return {"detected": False}

    try:
        signatures = [
            compute_screen_phash_signature(transition.evidence.screenshot)
            for transition in recent
        ]
    except (OSError, ValueError) as exc:
        logger.debug("transition cycle pHash check skipped", error=str(exc))
        return {"detected": False}
    hashes = [str(item.get("phash") or "") for item in signatures]
    sizes = [tuple(item.get("size") or []) for item in signatures]
    if not all(hashes) or len(set(sizes)) != 1:
        return {"detected": False}

    same_a = hamming_distance(hashes[0], hashes[2])
    same_b = hamming_distance(hashes[1], hashes[3])
    adjacent = [
        hamming_distance(hashes[index], hashes[index + 1]) for index in range(3)
    ]

    max_distance = get_settings().reflex.transition_cycle_phash_max_distance
    detected = bool(
        same_a is not None
        and same_b is not None
        and same_a <= max_distance
        and same_b <= max_distance
        and all(
            distance is not None and distance > max_distance for distance in adjacent
        )
    )
    if not detected:
        return {"detected": False}

    action_cycle: list[str] = []
    for transition in recent[:2]:
        action = transition.actions[0]
        detail = action.param.key or action.component
        action_cycle.append(f"{action.action}:{detail}" if detail else action.action)
    return {
        "detected": True,
        "action_cycle": action_cycle,
        "same_screen_distances": [same_a, same_b],
        "adjacent_screen_distances": adjacent,
    }


def latest_no_effect_transition(state: WorkerState) -> dict[str, Any]:
    """현재 화면에서 효과가 없다고 확인된 가장 최근 물리 행동을 반환한다."""

    events = [
        ExecutionEvent.model_validate(event)
        for event in state["transition"].get("action_events", []) or []
    ]
    completed_events = [event for event in events if event.transition]
    if not completed_events:
        return {}
    latest_event = completed_events[-1]
    latest = latest_event.transition
    evidence = latest.evidence
    if evidence is None or evidence.status != "unknown":
        return {}
    if evidence.reason not in {
        "reflex_no_screen_change",
        "no_screen_change",
    }:
        return {}
    latest_screen = str(state["observation"].get("current_screenshot") or "")
    observed_screen = evidence.screenshot
    if latest_screen and observed_screen and latest_screen != observed_screen:
        return {}
    action = latest.actions[0]
    result_args = latest_event.result.get("args")
    return {
        "action": action.action,
        "step": {
            "args": (
                dict(result_args)
                if isinstance(result_args, dict)
                else action.param.model_dump(
                    mode="json",
                    exclude_defaults=True,
                    exclude_none=True,
                )
            )
        },
        "status": evidence.status,
        "reason": evidence.reason,
        "screenshot": observed_screen,
    }


def transition_visual_change_ratio(
    transition_request: TransitionRequest,
    current_image_path: str | os.PathLike,
) -> float | None:
    """행동 전후 스크린샷의 눈에 띄는 픽셀 변화 비율을 계산한다."""

    before_image_path = str(transition_request.get("before_screenshot") or "")
    if not before_image_path or not current_image_path:
        return None
    try:
        from agent.vision.frame_compare import (
            changed_pixel_ratio,
            load_gray_frame,
        )

        intensity_threshold = get_settings().reflex.visual_change_pixel_threshold
        return changed_pixel_ratio(
            load_gray_frame(before_image_path),
            load_gray_frame(current_image_path),
            intensity_threshold=intensity_threshold,
        )
    except (OSError, ValueError) as exc:
        logger.debug("transition visual change check skipped", error=str(exc))
        return None


def transition_has_visual_change(
    transition_request: TransitionRequest,
    current_image_path: str | os.PathLike,
) -> tuple[bool, float | None]:
    """OpenCV 전후 프레임 비교로 화면 변화 시작 여부를 확인한다."""

    ratio = transition_visual_change_ratio(
        transition_request,
        current_image_path,
    )
    minimum_ratio = get_settings().reflex.visual_change_min_ratio
    return ratio is not None and ratio >= max(0.0, minimum_ratio), ratio


def idempotent_control_components() -> set[str]:
    return set(get_settings().reflex.idempotent_control_components)


def idempotent_page_scope_ignored_query_keys() -> set[str]:
    return set(get_settings().reflex.idempotent_scope_ignored_query_keys)


def idempotent_page_scope(url: str) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """탭 같은 UI 상태 쿼리를 제외하고 같은 페이지 범위를 계산한다."""

    if not url:
        return ("", "", ())
    try:
        parsed = urlparse(url)
        ignored = idempotent_page_scope_ignored_query_keys()
        query_items = tuple(
            sorted(
                (key.casefold(), value)
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                if key.casefold() not in ignored
            )
        )
        return ((parsed.netloc or "").casefold(), parsed.path or "/", query_items)
    except ValueError:
        return ("", url, ())


def same_idempotent_page_scope(left_url: str, right_url: str) -> bool:
    return bool(
        left_url
        and right_url
        and idempotent_page_scope(left_url) == idempotent_page_scope(right_url)
    )


def used_idempotent_recipe_keys_on_url(
    state: WorkerState, current_url: str
) -> set[str]:
    """같은 페이지 범위에서 이미 실행한 고정 UI recipe key를 찾는다."""

    if not current_url:
        return set()
    out: set[str] = set()
    components = idempotent_control_components()
    for action in action_event_results(
        state["transition"].get("action_events", []) or []
    ):
        if not isinstance(action, dict) or action.get("status") != "success":
            continue
        recipe_key = str(action.get("reflex_recipe_key") or "")
        if not recipe_key:
            continue
        before_url = str(action.get("before_url") or "")
        if before_url and not same_idempotent_page_scope(before_url, current_url):
            continue
        raw_args = action.get("args")
        args = raw_args if isinstance(raw_args, dict) else {}
        raw_target = action.get("target")
        target = raw_target if isinstance(raw_target, dict) else {}
        component = str(
            args.get("target_component") or target.get("component") or ""
        ).casefold()
        if component in components:
            out.add(recipe_key)
    return out


def transition_marker_texts(markers: list[dict[str, Any]]) -> list[str]:
    """전환 로그에는 중복을 제거한 OCR 텍스트만 남긴다."""

    seen: set[str] = set()
    texts: list[str] = []
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        value = normalize_text(marker.get("text"))
        key = value.casefold().replace(" ", "")
        if len(key) < 2 or key in seen or value.startswith("상호작용 가능한 요소"):
            continue
        seen.add(key)
        texts.append(value)
    return texts


def reused_ocr_observation(
    state: WorkerState,
    request: TransitionRequest,
) -> ObservationState:
    """화면 변화가 없을 때 직전 캡처의 OCR을 현재 관찰에 다시 연결한다."""

    observation = state["observation"]
    previous = dict(observation.get("previous_observation") or {})
    if not previous:
        return {}
    if str(request.get("before_observation_id") or "") != str(
        previous.get("observation_id") or ""
    ):
        return {}
    if str(request.get("before_screenshot") or "") != str(
        previous.get("screenshot") or ""
    ):
        return {}
    before_url = str(request.get("before_url") or "")
    previous_url = str(previous.get("current_url") or "")
    if before_url and previous_url and before_url != previous_url:
        return {}

    markers = [
        dict(marker)
        for marker in previous.get("markers", []) or []
        if isinstance(marker, dict)
    ]
    if not markers:
        return {}
    signature = dict(previous.get("screen_signature") or {})
    raw_signature = dict(observation.get("raw_screen_signature") or {})
    for key in ("phash", "size"):
        if raw_signature.get(key):
            signature[key] = raw_signature[key]
    current_observation = {
        **previous,
        "observation_id": str(observation.get("observation_id") or ""),
        "screenshot": str(observation.get("current_screenshot") or ""),
        "current_url": str(observation.get("current_url") or previous_url),
        "markers": markers,
        "screen_signature": signature,
    }
    return {
        "current_markers": markers,
        "ui_context": str(previous.get("ui_context") or ""),
        "marked_image": str(previous.get("marked_image") or ""),
        "screen_signature": signature,
        "current_page_role": str(previous.get("page_role") or ""),
        "ocr_complete": True,
        "previous_observation": current_observation,
    }


def transition_result(
    request: TransitionRequest | None,
    *,
    status: str,
    reason: str = "",
    visual_change_detected: bool = False,
    visual_change_ratio: float | None = None,
    needs_ocr: bool = False,
) -> TransitionResult:
    result: TransitionResult = {}
    if request is not None:
        result.update(request)
    result.update(
        {
            "status": status,
            "outcome": "",
            "reason": reason,
            "visual_change_detected": visual_change_detected,
            "visual_change_ratio": visual_change_ratio,
            "needs_ocr": needs_ocr,
        }
    )
    return result


def transition_result_without_request(
    state: WorkerState,
    request: TransitionRequest | None,
) -> dict[str, Any] | None:
    if state["observation"].get("low_information_screen"):
        return {
            "transition": {
                "transition_result": transition_result(
                    request,
                    status="pending" if request else "idle",
                    reason="low_information_screen",
                ),
            }
        }
    if request is None:
        return {
            "transition": {
                "transition_result": transition_result(
                    None,
                    status="idle",
                    reason="no_transition_request",
                    needs_ocr=not bool(state["observation"].get("ocr_complete")),
                ),
            }
        }
    return None


def input_text_confirmed_by_ocr(
    state: WorkerState,
    request: TransitionRequest,
) -> bool:
    """입력 문자열이 이전 화면에는 없고 현재 OCR에 나타났는지 확인한다."""

    if request.get("action") != "type_in_marker":
        return False
    input_text = str(request.get("input_text") or "").casefold().replace(" ", "")
    if not input_text:
        return False
    observation = state["observation"]
    previous = observation.get("previous_observation") or {}
    previous_markers = list(previous.get("markers") or [])
    previous_texts = transition_marker_texts(previous_markers)
    current_texts = transition_marker_texts(
        list(observation.get("current_markers") or [])
    )

    def contains_input(texts: list[str]) -> bool:
        return any(input_text in text.casefold().replace(" ", "") for text in texts)

    return contains_input(current_texts) and not contains_input(previous_texts)


def transition_record(
    request: TransitionRequest,
    *,
    status: str,
    source: str,
    reason: str,
    attempt: int,
    state: WorkerState,
    visual_change_ratio: float | None,
    ocr_skipped: bool,
) -> CompletedTransitionObservation:
    started_at = float(request.get("started_at") or time.time())
    observation = state["observation"]
    return build_transition_observation(
        request,
        status=status,
        outcome="",
        source=source,
        reason=reason,
        elapsed_sec=max(0.0, time.time() - started_at),
        attempt=attempt,
        markers=list(observation.get("current_markers", []) or []),
        screenshot=str(observation.get("current_screenshot") or ""),
        marked_image=str(observation.get("marked_image") or ""),
        after_observation_id=str(observation.get("observation_id") or ""),
        current_url=str(observation.get("current_url") or ""),
        page_role=str(observation.get("current_page_role") or ""),
        screen_signature=dict(observation.get("screen_signature") or {}),
        visual_change_ratio=visual_change_ratio,
        ocr_skipped=ocr_skipped,
    )


def build_transition_observation(
    transition_request: TransitionRequest,
    *,
    status: str,
    outcome: str,
    source: str,
    reason: str,
    elapsed_sec: float,
    attempt: int,
    markers: list[dict[str, Any]],
    screenshot: str,
    marked_image: str,
    after_observation_id: str = "",
    current_url: str = "",
    page_role: str = "",
    screen_signature: dict[str, Any] | None = None,
    phash_distance: int | None = None,
    visual_change_ratio: float | None = None,
    ocr_skipped: bool = False,
) -> CompletedTransitionObservation:
    """행동과 관찰 결과를 완성된 전환 근거로 만든다."""

    after_state = {
        "observation_id": str(after_observation_id or ""),
        "url_template": url_template(str(current_url or "")),
        "page_role": str(page_role or ""),
        "screen_context_signature": compact_screen_context_signature(screen_signature),
    }
    return CompletedTransitionObservation(
        action_seq=transition_request.get("action_seq"),
        action=transition_request.get("action", ""),
        before_observation_id=str(
            transition_request.get("before_observation_id") or ""
        ),
        after_observation_id=str(after_observation_id or ""),
        expected_after=str(transition_request.get("expected_after") or ""),
        source=source,
        recipe_key=transition_request.get("recipe_key", ""),
        recipe_transition_index=transition_request.get("recipe_transition_index"),
        recipe_transition_count=transition_request.get("recipe_transition_count"),
        transition_actions=list(transition_request.get("transition_actions") or []),
        after_state_match=dict(transition_request.get("after_state_match") or {}),
        attempt=attempt,
        elapsed_sec=round(elapsed_sec, 3),
        status=status,
        outcome=outcome,
        reason=reason,
        phash_distance=phash_distance,
        visual_change_ratio=visual_change_ratio,
        ocr_skipped=ocr_skipped,
        marker_count=len(markers),
        marker_texts=transition_marker_texts(markers),
        screenshot=str(screenshot),
        marked_image=str(marked_image or ""),
        current_url=str(current_url or ""),
        page_role=str(page_role or ""),
        after_state=after_state,
    )


__all__ = [
    "build_transition_observation",
    "detect_two_screen_transition_cycle",
    "idempotent_control_components",
    "idempotent_page_scope",
    "idempotent_page_scope_ignored_query_keys",
    "latest_no_effect_transition",
    "input_text_confirmed_by_ocr",
    "reused_ocr_observation",
    "same_idempotent_page_scope",
    "transition_has_visual_change",
    "transition_marker_texts",
    "transition_record",
    "transition_result",
    "transition_result_without_request",
    "used_idempotent_recipe_keys_on_url",
]
