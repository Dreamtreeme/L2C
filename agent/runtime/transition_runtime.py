"""행동 전후 화면 전환을 pHash와 관찰 기록으로 검증한다."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import parse_qsl, urlparse

from agent.config import get_settings
from agent.graph.state import GraphState
from agent.utils.logger import logger


def raw_screen_phash_signature(image_path: str | os.PathLike) -> dict[str, Any]:
    """OCR 없이 원본 스크린샷의 pHash와 크기만 계산한다."""

    try:
        from agent.vision.screen_signature import image_dimensions, perceptual_hash

        size = image_dimensions(image_path)
        return {
            "algorithm": "phash-dct64-v1",
            "phash": perceptual_hash(image_path),
            "size": list(size),
        }
    except Exception as exc:
        logger.debug("raw screen phash skipped", error=str(exc))
        return {"algorithm": "phash-dct64-v1", "phash": "", "size": [0, 0]}


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

    signatures = [
        raw_screen_phash_signature(str(item.get("screenshot") or ""))
        for item in recent
    ]
    hashes = [str(item.get("phash") or "") for item in signatures]
    sizes = [tuple(item.get("size") or []) for item in signatures]
    if not all(hashes) or len(set(sizes)) != 1:
        return {"detected": False}

    try:
        from agent.vision.screen_signature import hamming_distance

        same_a = hamming_distance(hashes[0], hashes[2])
        same_b = hamming_distance(hashes[1], hashes[3])
        adjacent = [
            hamming_distance(hashes[index], hashes[index + 1])
            for index in range(3)
        ]
    except Exception as exc:
        logger.debug("transition cycle pHash check skipped", error=str(exc))
        return {"detected": False}

    max_distance = get_settings().reflex.transition_cycle_phash_max_distance
    detected = bool(
        same_a is not None
        and same_b is not None
        and same_a <= max_distance
        and same_b <= max_distance
        and all(distance is not None and distance > max_distance for distance in adjacent)
    )
    if not detected:
        return {"detected": False}

    action_cycle: list[str] = []
    for observation in recent[:2]:
        action = str(observation.get("action") or "")
        step = observation.get("step") if isinstance(observation.get("step"), dict) else {}
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        detail = str(args.get("key") or args.get("target_component") or "")
        action_cycle.append(f"{action}:{detail}" if detail else action)
    return {
        "detected": True,
        "action_cycle": action_cycle,
        "same_screen_distances": [same_a, same_b],
        "adjacent_screen_distances": adjacent,
    }


def latest_no_effect_transition(state: GraphState) -> dict[str, Any]:
    """현재 화면에서 효과가 없다고 확인된 가장 최근 물리 행동을 반환한다."""

    observations = state.get("transition_observations", []) or []
    if not observations:
        return {}
    latest = observations[-1]
    if not isinstance(latest, dict) or latest.get("status") != "unknown":
        return {}
    if latest.get("reason") not in {"reflex_no_screen_change", "no_screen_change"}:
        return {}
    recent_images = state.get("recent_images", []) or []
    latest_screen = str(recent_images[-1]) if recent_images else ""
    observed_screen = str(latest.get("screenshot") or "")
    if latest_screen and observed_screen and latest_screen != observed_screen:
        return {}
    return latest


def transition_visual_change_ratio(
    pending_transition: dict[str, Any],
    current_image_path: str | os.PathLike,
) -> float | None:
    """행동 전후 스크린샷의 눈에 띄는 픽셀 변화 비율을 계산한다."""

    before_image_path = str(pending_transition.get("before_screenshot") or "")
    if not before_image_path or not current_image_path:
        return None
    try:
        from PIL import Image, ImageChops

        target_size = (196, 212)
        with Image.open(before_image_path) as before_image, Image.open(current_image_path) as current_image:
            before = before_image.convert("L").resize(target_size)
            current = current_image.convert("L").resize(target_size)
            histogram = ImageChops.difference(before, current).histogram()
        intensity_threshold = get_settings().reflex.visual_change_pixel_threshold
        intensity_threshold = min(255, max(0, intensity_threshold))
        changed_pixels = sum(histogram[intensity_threshold + 1 :])
        return changed_pixels / float(target_size[0] * target_size[1])
    except Exception as exc:
        logger.debug("transition visual change check skipped", error=str(exc))
        return None


def transition_has_visual_change(
    pending_transition: dict[str, Any],
    current_image_path: str | os.PathLike,
) -> tuple[bool, float | None]:
    """전체 pHash가 둔감한 부분 화면 전환을 전후 스크린샷으로 보완한다."""

    ratio = transition_visual_change_ratio(pending_transition, current_image_path)
    minimum_ratio = get_settings().reflex.visual_change_min_ratio
    return ratio is not None and ratio >= max(0.0, minimum_ratio), ratio


def transition_no_effect_by_phash(
    pending_transition: dict[str, Any],
    current_url: str,
    raw_screen_signature: dict[str, Any],
) -> tuple[bool, int | None]:
    """OCR 전에 같은 URL과 거의 같은 pHash면 행동 효과 없음으로 판정한다."""

    if not pending_transition:
        return False, None
    source = str(pending_transition.get("source") or "")
    action = str(pending_transition.get("action") or "")
    if source not in {"reflex", "page_policy", "card_queue", "autonomous"}:
        return False, None
    if action not in {
        "click_marker",
        "press_key",
        "go_back",
        "close_current_tab",
        "switch_tab",
    }:
        return False, None
    before_url = str(pending_transition.get("before_url") or "")
    before_phash = str(pending_transition.get("before_phash") or "")
    current_phash = str(raw_screen_signature.get("phash") or "")
    if not before_url or not current_url or before_url != current_url:
        return False, None
    if not before_phash or not current_phash:
        return False, None
    try:
        from agent.vision.screen_signature import hamming_distance

        distance = hamming_distance(before_phash, current_phash)
    except Exception as exc:
        logger.debug("transition phash no-effect check skipped", error=str(exc))
        return False, None
    max_distance = get_settings().reflex.no_effect_phash_max_distance
    return distance is not None and distance <= max_distance, distance


def transition_phash_distance(
    pending_transition: dict[str, Any],
    current_url: str,
    screen_signature: dict[str, Any],
) -> tuple[bool, int | None, int]:
    """같은 URL에서 전환 전후 pHash 거리를 계산한다."""

    before_url = str(pending_transition.get("before_url") or "")
    before_phash = str(pending_transition.get("before_phash") or "")
    current_phash = str(screen_signature.get("phash") or "")
    same_url = bool(before_url and current_url and before_url == current_url)
    max_distance = get_settings().reflex.no_effect_phash_max_distance
    if not same_url or not before_phash or not current_phash:
        return same_url, None, max_distance
    try:
        from agent.vision.screen_signature import hamming_distance

        return same_url, hamming_distance(before_phash, current_phash), max_distance
    except Exception as exc:
        logger.debug("transition phash distance skipped", error=str(exc))
        return same_url, None, max_distance


def visual_change_sufficient_components() -> set[str]:
    return set(get_settings().reflex.visual_change_sufficient_components)


def transition_accepts_visual_change(pending_transition: dict[str, Any]) -> bool:
    """일부 UI step은 OCR cue보다 pHash 변화로 성공을 판정한다."""

    source = str(pending_transition.get("source") or "")
    if source not in {"reflex", "page_policy"}:
        return False
    if str(pending_transition.get("action") or "") != "click_marker":
        return False
    step = dict(pending_transition.get("step", {}) or {})
    args = step.get("args") if isinstance(step.get("args"), dict) else {}
    labels = {
        str(step.get("component") or "").casefold(),
        str(step.get("target_role") or "").casefold(),
        str(args.get("target_component") or "").casefold(),
        str(args.get("target_role") or "").casefold(),
    }
    return bool(labels & visual_change_sufficient_components())


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
    except Exception:
        return ("", url, ())


def same_idempotent_page_scope(left_url: str, right_url: str) -> bool:
    return bool(left_url and right_url and idempotent_page_scope(left_url) == idempotent_page_scope(right_url))


def used_idempotent_recipe_keys_on_url(state: GraphState, current_url: str) -> set[str]:
    """같은 페이지 범위에서 이미 실행한 고정 UI recipe key를 찾는다."""

    if not current_url:
        return set()
    out: set[str] = set()
    components = idempotent_control_components()
    for action in state.get("action_history", []) or []:
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
        component = str(args.get("target_component") or target.get("component") or "").casefold()
        if component in components:
            out.add(recipe_key)
    return out


def transition_marker_texts(markers: list[dict[str, Any]]) -> list[str]:
    try:
        from agent.recipe.transition import marker_texts

        return marker_texts(markers)
    except Exception:
        return []


def build_transition_observation(
    pending_transition: dict[str, Any],
    *,
    transition_status: str,
    transition_outcome: str,
    transition_source: str,
    reason: str,
    elapsed_sec: float,
    attempt: int,
    markers: list[dict[str, Any]],
    screenshot: str,
    marked_image: str,
    to_capture_id: str = "",
    phash_distance: int | None = None,
    visual_change_ratio: float | None = None,
    ocr_skipped: bool = False,
) -> dict[str, Any]:
    """행동 step과 관찰 결과를 한 묶음으로 만든다."""

    return {
        "action_seq": pending_transition.get("action_seq"),
        "action": pending_transition.get("action", ""),
        "from_capture_id": str(pending_transition.get("from_capture_id") or ""),
        "to_capture_id": str(to_capture_id or ""),
        "step": dict(pending_transition.get("step", {}) or {}),
        "expected_after": pending_transition.get("expected_after", ""),
        "source": transition_source,
        "recipe_key": pending_transition.get("recipe_key", ""),
        "attempt": attempt,
        "elapsed_sec": round(elapsed_sec, 3),
        "status": transition_status,
        "outcome": transition_outcome,
        "reason": reason,
        "phash_distance": phash_distance,
        "visual_change_ratio": visual_change_ratio,
        "ocr_skipped": ocr_skipped,
        "marker_count": len(markers),
        "marker_texts": transition_marker_texts(markers),
        "screenshot": str(screenshot),
        "marked_image": str(marked_image or ""),
    }


__all__ = [
    "build_transition_observation",
    "detect_two_screen_transition_cycle",
    "idempotent_control_components",
    "idempotent_page_scope",
    "idempotent_page_scope_ignored_query_keys",
    "latest_no_effect_transition",
    "raw_screen_phash_signature",
    "same_idempotent_page_scope",
    "transition_accepts_visual_change",
    "transition_has_visual_change",
    "transition_marker_texts",
    "transition_no_effect_by_phash",
    "transition_phash_distance",
    "used_idempotent_recipe_keys_on_url",
    "visual_change_sufficient_components",
]
