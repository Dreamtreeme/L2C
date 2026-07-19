"""검색 결과 카드 큐를 만들고 뒤로가기 후 다음 카드를 재생한다."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage

from agent.graph.action_request import build_action_message
from agent.graph.state import GraphState
from agent.runtime.site_context import looks_like_job_detail_url
from agent.vision.marker_geometry import (
    bbox_from_ratio,
    bbox_to_ratio,
    center_ratio_from_bbox,
    screen_size_from_signature,
)


def marker_by_id(markers: list[dict], marker_id: int | None) -> dict | None:
    for marker in markers or []:
        if isinstance(marker, dict) and marker.get("id") == marker_id:
            return marker
    return None


def card_queue_enabled() -> bool:
    raw = os.getenv("VISION_RESULT_CARD_QUEUE_ENABLED", "1")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def queue_card_label(card: dict) -> str:
    for key in ("title", "target_label", "position", "text", "label"):
        value = card.get(key)
        if value not in (None, "", [], {}):
            return str(value).strip()
    return ""


def text_list_arg(args: dict, key: str) -> list[str]:
    value = args.get(key)
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := str(item or "").strip())]


def result_card_entries_from_args(args: dict) -> list[dict]:
    """도구 입력을 카드 후보 목록으로 통일한다."""

    cards = args.get("cards")
    if isinstance(cards, list):
        entries: list[dict] = []
        for raw in cards:
            if isinstance(raw, dict):
                entries.append(dict(raw))
            elif isinstance(raw, str) and raw.strip():
                entries.append({"title": raw.strip()})
        if entries:
            return entries

    titles = text_list_arg(args, "titles") or text_list_arg(args, "target_labels")
    companies = text_list_arg(args, "companies")
    entries = []
    for index, title in enumerate(titles):
        entry = {"title": title}
        if index < len(companies):
            entry["company"] = companies[index]
        entries.append(entry)
    return entries


def queue_match_text(value: Any) -> str:
    try:
        from agent.recipe.text_utils import normalize_text

        text = normalize_text(value)
    except Exception:
        text = str(value or "").strip()
    return text.casefold().replace(" ", "")


def marker_by_label(markers: list[dict], label: str, used_marker_ids: set[int]) -> dict | None:
    label_key = queue_match_text(label)
    if not label_key:
        return None
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        try:
            marker_id = int(marker.get("id"))
        except (TypeError, ValueError):
            marker_id = -1
        if marker_id in used_marker_ids:
            continue
        if queue_match_text(marker.get("text")) == label_key:
            return marker
    return None


def _target_count_from_state(state: GraphState) -> int:
    params = state.get("recipe_params", {}) or {}
    try:
        return max(0, int(params.get("target_count") or 0))
    except (TypeError, ValueError):
        return 0


def _collected_job_count(extracted_jd: Any) -> int:
    if not isinstance(extracted_jd, dict) or not extracted_jd:
        return 0
    for value in extracted_jd.values():
        if isinstance(value, list) and any(isinstance(item, dict) and item for item in value):
            return sum(1 for item in value if isinstance(item, dict) and item)
    return 1


def normalize_result_card_queue(args: dict, state: GraphState, current_url: str) -> tuple[list[dict], dict]:
    """LLM이 고른 현재 화면의 공고 카드를 좌표비율 기반 큐로 정규화한다."""

    cards = result_card_entries_from_args(args)
    markers = list(state.get("current_markers", []) or [])
    signature = dict(state.get("screen_signature", {}) or {})
    size = screen_size_from_signature(signature)
    queue: list[dict] = [
        dict(item)
        for item in (state.get("result_card_queue", []) or [])
        if isinstance(item, dict) and str(item.get("status") or "") != "pending"
    ]
    existing_labels = {
        (queue_match_text(item.get("title")), queue_match_text(item.get("company")))
        for item in queue
    }
    used_marker_ids: set[int] = set()
    target_count = _target_count_from_state(state)
    resolved_count = max(
        _collected_job_count(state.get("extracted_jd", {}) or {}),
        completed_result_card_count(queue),
    )
    remaining = (
        target_count - resolved_count
        if target_count > 0
        else len(cards)
    )
    limit = max(0, remaining) if target_count > 0 else len(cards)

    for raw in cards[:limit]:
        if not isinstance(raw, dict):
            continue
        marker_id = raw.get("marker_id", raw.get("id"))
        try:
            marker_id = int(marker_id) if marker_id is not None else None
        except (TypeError, ValueError):
            marker_id = None
        marker = marker_by_id(markers, marker_id)
        label = queue_card_label(raw) or (str(marker.get("text") or "").strip() if marker else "")
        if marker is None and label:
            marker = marker_by_label(markers, label, used_marker_ids)
            if marker:
                try:
                    marker_id = int(marker.get("id"))
                except (TypeError, ValueError):
                    marker_id = None
        if not label:
            continue
        if marker_id is not None:
            used_marker_ids.add(marker_id)

        bbox = marker.get("bbox") if marker else raw.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = []
        bbox_ratio = raw.get("bbox_ratio")
        center_ratio = raw.get("center_ratio")
        if marker and size and not bbox_ratio:
            try:
                bbox_ratio = bbox_to_ratio(bbox, size)
                center_ratio = center_ratio_from_bbox(bbox, size)
            except Exception:
                bbox_ratio = []
                center_ratio = []
        if (not bbox_ratio or len(bbox_ratio) != 4) and raw.get("bbox_ratio"):
            bbox_ratio = raw.get("bbox_ratio")
        if (not center_ratio or len(center_ratio) != 2) and raw.get("center_ratio"):
            center_ratio = raw.get("center_ratio")
        if not bbox_ratio and not marker:
            continue

        company = str(raw.get("company") or raw.get("company_name") or "").strip()
        identity = (queue_match_text(label), queue_match_text(company))
        if identity in existing_labels or (identity[0], "") in existing_labels:
            continue
        existing_labels.add(identity)

        queue_id = str(raw.get("queue_id") or f"card-{len(queue) + 1}")
        existing_queue_ids = {str(item.get("queue_id") or "") for item in queue}
        if queue_id in existing_queue_ids:
            queue_id = f"card-{len(queue) + 1}"
        evidence_texts = raw.get("evidence_texts") if isinstance(raw.get("evidence_texts"), list) else []
        if company and company not in evidence_texts:
            evidence_texts = [company, *evidence_texts]
        queue.append(
            {
                "queue_id": queue_id,
                "status": "pending",
                "title": label,
                "company": company,
                "source_marker_id": marker_id,
                "bbox_ratio": bbox_ratio or [],
                "center_ratio": center_ratio or [],
                "evidence_texts": evidence_texts[:6],
                "target": {
                    "text": str(marker.get("text") or label) if marker else label,
                    "semantic_label": label,
                    "bbox_ratio": bbox_ratio or [],
                    "center_ratio": center_ratio or [],
                    "evidence_texts": evidence_texts[:6],
                },
            }
        )

    memory = {
        "url": current_url or state.get("current_url", "") or "",
        "screen_signature": signature,
        "screenshot": str((state.get("recent_images") or [""])[-1] or "") if state.get("recent_images") else "",
        "marked_image": state.get("marked_image", "") or "",
    }
    return queue, memory


def pending_result_cards(queue: list[dict]) -> list[dict]:
    return [
        dict(item)
        for item in queue or []
        if isinstance(item, dict) and str(item.get("status") or "pending") == "pending"
    ]


def terminal_result_card_count(queue: list[dict]) -> int:
    """수집 완료 또는 DB 중복으로 현재 큐 처리가 끝난 카드 수를 반환한다."""

    return sum(
        1
        for item in queue or []
        if isinstance(item, dict) and str(item.get("status") or "") in {"done", "skipped"}
    )


def completed_result_card_count(queue: list[dict]) -> int:
    """상세 정보를 실제로 수집 완료한 카드 수만 반환한다."""

    return sum(
        1
        for item in queue or []
        if isinstance(item, dict) and str(item.get("status") or "") == "done"
    )


def result_card_queue_scope_complete(
    queue: list[dict],
    *,
    count_mode: str,
    target_count: int,
) -> bool:
    """현재 검색 결과 큐가 사용자 요청 범위를 충족했는지 판단한다."""

    if not queue or any(
        str(item.get("status") or "pending") in {"pending", "active"}
        for item in queue
        if isinstance(item, dict)
    ):
        return False
    if str(count_mode or "").strip().lower() == "visible_all":
        return True
    return target_count > 0 and completed_result_card_count(queue) >= target_count


def same_queue_card(item: dict, args: dict) -> bool:
    if args.get("queue_id") and str(args.get("queue_id")) == str(item.get("queue_id")):
        return True
    label = str(args.get("target_label") or "").strip()
    if label and label == str(item.get("title") or "").strip():
        return True
    marker_id = args.get("marker_id")
    return marker_id is not None and str(marker_id) == str(item.get("source_marker_id"))


def mark_result_card_active(queue: list[dict], args: dict) -> tuple[list[dict], dict]:
    updated = []
    active: dict = {}
    for raw in queue or []:
        item = dict(raw)
        if not active and item.get("status") == "pending" and same_queue_card(item, args):
            item["status"] = "active"
            active = dict(item)
        updated.append(item)
    return updated, active


def result_card_click_matches_queue(queue: list[dict], args: dict) -> bool:
    if not queue:
        return False
    if args.get("queue_id"):
        return True
    component = str(args.get("target_component") or "")
    role = str(args.get("target_role") or "")
    if component in {"job_card", "job_card_title"} or role in {"job_card", "job_card_title"}:
        return True
    label = str(args.get("target_label") or "").strip()
    return bool(label) and any(
        label == str(item.get("title") or "").strip()
        for item in queue
        if isinstance(item, dict)
    )


def complete_active_result_card(queue: list[dict], active_card: dict) -> tuple[list[dict], dict]:
    if not active_card:
        return queue, {}
    active_id = str(active_card.get("queue_id") or "")
    updated = []
    for raw in queue or []:
        item = dict(raw)
        if active_id and str(item.get("queue_id") or "") == active_id:
            item["status"] = "done"
        updated.append(item)
    return updated, {}


def skip_active_result_card(
    queue: list[dict],
    active_card: dict,
    *,
    reason: str,
    url: str,
    job_id: int | None = None,
) -> tuple[list[dict], dict]:
    """이미 수집한 상세 URL의 활성 카드를 DB 근거가 확인된 상태로 끝낸다."""

    if not active_card:
        return queue, {}
    active_id = str(active_card.get("queue_id") or "")
    updated = []
    for raw in queue or []:
        item = dict(raw)
        if active_id and str(item.get("queue_id") or "") == active_id:
            item.update({"status": "skipped", "skip_reason": reason, "detail_url": url})
            if job_id is not None:
                item["job_id"] = int(job_id)
        updated.append(item)
    return updated, {}


def queue_return_screen_matches(
    memory: dict,
    current_url: str,
    current_signature: dict,
    *,
    require_anchors: bool = True,
) -> tuple[bool, dict]:
    if not memory:
        return False, {"reason": "queue_memory_missing"}
    if looks_like_job_detail_url(current_url):
        return False, {"reason": "still_on_detail_url", "url": current_url}

    saved_signature = dict(memory.get("screen_signature") or {})
    saved_phash = str(saved_signature.get("phash") or "")
    current_phash = str((current_signature or {}).get("phash") or "")
    if not saved_phash or not current_phash:
        return False, {
            "reason": "phash_missing",
            "saved_phash": bool(saved_phash),
            "current_phash": bool(current_phash),
        }
    saved_size = list(saved_signature.get("size") or [])
    current_size = list((current_signature or {}).get("size") or [])
    if saved_size and current_size and saved_size != current_size:
        return False, {
            "reason": "capture_size_mismatch",
            "saved_size": saved_size,
            "current_size": current_size,
        }
    try:
        from agent.recipe.phash_replay import anchor_overlap
        from agent.vision.screen_signature import hamming_distance

        distance = hamming_distance(saved_phash, current_phash)
        overlap = (
            anchor_overlap(
                saved_signature.get("anchors") or [],
                (current_signature or {}).get("anchors") or [],
            )
            if require_anchors
            else None
        )
    except Exception as exc:
        return False, {"reason": "phash_compare_failed", "error": str(exc)}

    try:
        max_distance = int(os.getenv("VISION_CARD_QUEUE_RETURN_PHASH_MAX_DISTANCE", "16"))
        min_overlap = float(os.getenv("VISION_CARD_QUEUE_RETURN_MIN_ANCHOR_OVERLAP", "0.20"))
    except ValueError:
        max_distance = 16
        min_overlap = 0.20
    matched = bool(
        distance is not None
        and distance <= max_distance
        and (not require_anchors or (overlap is not None and overlap >= min_overlap))
    )
    return matched, {
        "reason": (
            "phash_anchor_match"
            if matched and require_anchors
            else "phash_match"
            if matched
            else "phash_anchor_mismatch"
            if require_anchors
            else "phash_mismatch"
        ),
        "distance": distance,
        "max_distance": max_distance,
        "anchor_overlap": overlap,
        "min_anchor_overlap": min_overlap,
    }


def queue_marker_for_item(item: dict, markers: list[dict], signature: dict) -> tuple[int | None, list[dict], dict]:
    target = dict(item.get("target") or {})
    target.setdefault("text", item.get("title", ""))
    target.setdefault("semantic_label", item.get("title", ""))
    target.setdefault("bbox_ratio", item.get("bbox_ratio") or [])
    target.setdefault("center_ratio", item.get("center_ratio") or [])
    target.setdefault("evidence_texts", item.get("evidence_texts") or [])
    try:
        from agent.recipe.phash_replay import match_target_by_ratio

        marker_id = match_target_by_ratio(target, markers, screen_size_from_signature(signature))
        if marker_id is not None:
            return marker_id, markers, {"reason": "current_marker_ratio_match"}
    except Exception as exc:
        ratio_error = str(exc)
    else:
        ratio_error = ""

    bbox = bbox_from_ratio(item.get("bbox_ratio") or [], screen_size_from_signature(signature))
    if bbox == [0, 0, 0, 0]:
        return None, markers, {"reason": "cached_bbox_missing", "ratio_error": ratio_error}
    next_id = max(
        [int(marker.get("id") or 0) for marker in markers or [] if isinstance(marker, dict)] + [-1]
    ) + 1
    synthetic = {
        "id": next_id,
        "bbox": bbox,
        "text": item.get("title") or "queued result card",
        "type": "queue_cached_card",
    }
    return next_id, [*markers, synthetic], {
        "reason": "synthetic_marker_from_cached_bbox",
        "bbox": bbox,
        "ratio_error": ratio_error,
    }


def queue_replay_after_return(
    state: GraphState,
    observed_transition: dict,
    current_url: str,
    markers: list[dict],
    screen_signature: dict,
    *,
    require_anchors: bool = True,
) -> tuple[AIMessage | None, list[dict], dict]:
    """어떤 물리 행동으로 복귀했든 목록 화면이 확인되면 다음 카드를 준비한다."""

    if not card_queue_enabled():
        return None, markers, {"reason": "queue_disabled"}
    if not str(observed_transition.get("action") or ""):
        return None, markers, {"reason": "return_transition_missing"}
    queue = [
        dict(item)
        for item in (state.get("result_card_queue", []) or [])
        if isinstance(item, dict)
    ]
    pending = pending_result_cards(queue)
    if not pending:
        return None, markers, {"reason": "queue_empty"}
    active_card = dict(state.get("active_result_card", {}) or {})
    if active_card:
        return None, markers, {
            "reason": "active_card_not_completed",
            "queue_id": active_card.get("queue_id", ""),
            "title": active_card.get("title", ""),
        }

    matched, match_trace = queue_return_screen_matches(
        dict(state.get("result_page_memory", {}) or {}),
        current_url,
        screen_signature,
        require_anchors=require_anchors,
    )
    if not matched:
        return None, markers, match_trace

    item = pending[0]
    marker_id, next_markers, marker_trace = queue_marker_for_item(item, markers, screen_signature)
    if marker_id is None:
        trace = dict(match_trace)
        trace.update(marker_trace)
        return None, markers, trace
    args = {
        "marker_id": marker_id,
        "queue_id": item.get("queue_id", ""),
        "target_label": item.get("title", ""),
        "target_role": "job_card",
        "target_component": "job_card_title",
        "reason": "검색 결과 카드 큐에서 다음 미방문 공고를 선택합니다.",
        "expected_after": "선택한 공고의 상세 페이지가 열린다.",
    }
    message = build_action_message(
        "card_queue",
        "cached next result card",
        [
            {
                "name": "click_marker",
                "args": args,
                "id": f"card_queue_{item.get('queue_id', 'next')}",
            }
        ],
    )
    return message, next_markers, {
        "hit": True,
        "queue_id": item.get("queue_id", ""),
        "title": item.get("title", ""),
        "return_match": match_trace,
        "marker": marker_trace,
    }


__all__ = [
    "card_queue_enabled",
    "completed_result_card_count",
    "complete_active_result_card",
    "mark_result_card_active",
    "marker_by_id",
    "marker_by_label",
    "normalize_result_card_queue",
    "pending_result_cards",
    "result_card_queue_scope_complete",
    "queue_card_label",
    "queue_marker_for_item",
    "queue_match_text",
    "queue_replay_after_return",
    "queue_return_screen_matches",
    "result_card_click_matches_queue",
    "result_card_entries_from_args",
    "same_queue_card",
    "skip_active_result_card",
    "terminal_result_card_count",
    "text_list_arg",
]
