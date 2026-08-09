"""행동 전후 프레임 변화와 저장 상태 검증에 필요한 관찰 기록을 만든다."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import parse_qsl, urlparse

from agent.config import get_settings
from agent.runtime.worker_contracts import (
    WorkerState,
    action_event_results,
    action_event_transitions,
)
from agent.utils.logger import logger
from agent.utils.text import normalize_text, url_template
from agent.vision.screen_signature import (
    compact_screen_context_signature,
    compute_screen_phash_signature,
    hamming_distance,
)


def detect_two_screen_transition_cycle(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """최근 전환 화면이 A-B-A-B로 반복됐는지 pHash로 확인한다."""

    recent = [
        item
        for item in observations or []
        if isinstance(item, dict) and str(item.get("screenshot") or "")
    ][-4:]
    if len(recent) < 4:
        return {"detected": False}

    try:
        signatures = [
            compute_screen_phash_signature(str(item.get("screenshot") or ""))
            for item in recent
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
    for observation in recent[:2]:
        action = str(observation.get("action") or "")
        step = (
            observation.get("step") if isinstance(observation.get("step"), dict) else {}
        )
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        detail = str(args.get("key") or args.get("target_component") or "")
        action_cycle.append(f"{action}:{detail}" if detail else action)
    return {
        "detected": True,
        "action_cycle": action_cycle,
        "same_screen_distances": [same_a, same_b],
        "adjacent_screen_distances": adjacent,
    }


def latest_no_effect_transition(state: WorkerState) -> dict[str, Any]:
    """현재 화면에서 효과가 없다고 확인된 가장 최근 물리 행동을 반환한다."""

    observations = action_event_transitions(
        state["transition"].get("action_events", []) or []
    )
    if not observations:
        return {}
    latest = observations[-1]
    if not isinstance(latest, dict) or latest.get("status") != "unknown":
        return {}
    if latest.get("reason") not in {"reflex_no_screen_change", "no_screen_change"}:
        return {}
    latest_screen = str(state["observation"].get("current_screenshot") or "")
    observed_screen = str(latest.get("screenshot") or "")
    if latest_screen and observed_screen and latest_screen != observed_screen:
        return {}
    return latest


def transition_visual_change_ratio(
    transition_request: dict[str, Any],
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
    transition_request: dict[str, Any],
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
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        target = action.get("target") if isinstance(action.get("target"), dict) else {}
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


def build_transition_observation(
    transition_request: dict[str, Any],
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
    to_capture_id: str = "",
    current_url: str = "",
    page_role: str = "",
    screen_signature: dict[str, Any] | None = None,
    phash_distance: int | None = None,
    visual_change_ratio: float | None = None,
    ocr_skipped: bool = False,
) -> dict[str, Any]:
    """행동 step과 관찰 결과를 한 묶음으로 만든다."""

    after_state = {
        "capture_id": str(to_capture_id or ""),
        "url_template": url_template(str(current_url or "")),
        "page_role": str(page_role or ""),
        "screen_context_signature": compact_screen_context_signature(screen_signature),
    }
    return {
        "action_seq": transition_request.get("action_seq"),
        "action": transition_request.get("action", ""),
        "from_capture_id": str(transition_request.get("from_capture_id") or ""),
        "to_capture_id": str(to_capture_id or ""),
        "step": dict(transition_request.get("step", {}) or {}),
        "expected_after": dict(transition_request.get("step") or {}).get(
            "expected_after", ""
        ),
        "source": source,
        "recipe_key": transition_request.get("recipe_key", ""),
        "recipe_transition_index": transition_request.get("recipe_transition_index"),
        "recipe_transition_count": transition_request.get("recipe_transition_count"),
        "transition_actions": list(transition_request.get("transition_actions") or []),
        "after_state_match": dict(transition_request.get("after_state_match") or {}),
        "attempt": attempt,
        "elapsed_sec": round(elapsed_sec, 3),
        "status": status,
        "outcome": outcome,
        "reason": reason,
        "phash_distance": phash_distance,
        "visual_change_ratio": visual_change_ratio,
        "ocr_skipped": ocr_skipped,
        "marker_count": len(markers),
        "marker_texts": transition_marker_texts(markers),
        "screenshot": str(screenshot),
        "marked_image": str(marked_image or ""),
        "current_url": str(current_url or ""),
        "page_role": str(page_role or ""),
        "after_state": after_state,
    }


__all__ = [
    "build_transition_observation",
    "detect_two_screen_transition_cycle",
    "idempotent_control_components",
    "idempotent_page_scope",
    "idempotent_page_scope_ignored_query_keys",
    "latest_no_effect_transition",
    "same_idempotent_page_scope",
    "transition_has_visual_change",
    "transition_marker_texts",
    "used_idempotent_recipe_keys_on_url",
]
