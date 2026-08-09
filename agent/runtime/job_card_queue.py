"""채용공고 카드 큐를 만들고 목록 복귀 후 다음 카드를 재생한다."""

from __future__ import annotations

from typing import Any

from agent.config import get_settings
from agent.runtime.worker_contracts import (
    ActionRequest,
    WorkerState,
    build_action_request,
)
from agent.runtime.target_matching import anchor_overlap, match_target_by_ratio
from agent.utils.text import normalize_text
from agent.runtime.job_collection import job_count
from agent.runtime.site_context import looks_like_job_detail_url
from agent.runtime.worker_state import target_count_from_state
from agent.vision.marker_geometry import (
    bbox_from_ratio,
    bbox_to_ratio,
    center_ratio_from_bbox,
    screen_size_from_signature,
)
from agent.vision.screen_signature import hamming_distance
from agent.vision.target_snapshot import marker_by_id


def job_card_label(card: dict) -> str:
    return str(card.get("title") or "").strip()


def job_card_entries_from_args(args: dict) -> list[dict]:
    """도구 입력을 카드 후보 목록으로 통일한다."""

    return [dict(card) for card in args.get("cards") or []]


def job_card_match_text(value: Any) -> str:
    text = normalize_text(value)
    return text.casefold().replace(" ", "")


def _queue_limit(
    state: WorkerState,
    queue: list[dict],
    card_count: int,
) -> int:
    target_count = target_count_from_state(state)
    if target_count <= 0:
        return card_count
    collection = state["collection"]
    resolved_count = max(
        job_count(collection.get("collected_jobs", [])),
        resolved_job_card_count(queue),
    )
    return max(0, target_count - resolved_count)


def _card_geometry(
    marker: dict[str, Any],
    size: list[int],
) -> tuple[Any, Any]:
    bbox = marker.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        bbox = []
    if not size:
        return [], []
    return bbox_to_ratio(bbox, size), center_ratio_from_bbox(bbox, size)


def _normalized_job_card(
    raw: dict[str, Any],
    markers: list[dict],
    size: list[int],
    queue: list[dict],
) -> dict[str, Any] | None:
    marker_id = int(raw["marker_id"])
    marker = marker_by_id(markers, marker_id)
    if marker is None:
        return None
    label = job_card_label(raw)
    if not label:
        return None
    bbox_ratio, center_ratio = _card_geometry(marker, size)
    company = str(raw.get("company") or "").strip()
    evidence_texts = [company] if company else []
    return {
        "queue_id": f"card-{len(queue) + 1}",
        "status": "pending",
        "title": label,
        "company": company,
        "source_marker_id": marker_id,
        "bbox_ratio": bbox_ratio,
        "center_ratio": center_ratio,
        "evidence_texts": evidence_texts,
        "target": {
            "text": str(marker.get("text") or label),
            "semantic_label": label,
            "bbox_ratio": bbox_ratio,
            "center_ratio": center_ratio,
            "evidence_texts": evidence_texts,
            "marker_type": str(marker.get("type") or ""),
        },
    }


def _job_results_memory(
    observation: dict[str, Any],
    collection: dict[str, Any],
    current_url: str,
) -> dict[str, Any]:
    memory = {
        "url": current_url or observation.get("current_url", "") or "",
        "screen_signature": dict(observation.get("screen_signature", {}) or {}),
        "screenshot": str(observation.get("current_screenshot") or ""),
        "marked_image": observation.get("marked_image", "") or "",
    }
    previous_return_action = dict(
        (collection.get("job_results_memory") or {}).get("return_action") or {}
    )
    if previous_return_action:
        memory["return_action"] = previous_return_action
    return memory


def normalize_job_card_queue(
    args: dict,
    state: WorkerState,
    current_url: str,
) -> tuple[list[dict], dict]:
    """LLM이 고른 현재 화면의 공고 카드를 좌표비율 기반 큐로 정규화한다."""

    cards = job_card_entries_from_args(args)
    observation = state["observation"]
    collection = state["collection"]
    markers = list(observation.get("current_markers", []) or [])
    size = screen_size_from_signature(
        dict(observation.get("screen_signature", {}) or {})
    )
    queue: list[dict] = [
        dict(item)
        for item in (collection.get("job_card_queue", []) or [])
        if isinstance(item, dict) and str(item.get("status") or "") != "pending"
    ]
    existing_labels = {
        (
            job_card_match_text(item.get("title")),
            job_card_match_text(item.get("company")),
        )
        for item in queue
    }
    for raw in cards[: _queue_limit(state, queue, len(cards))]:
        card = _normalized_job_card(raw, markers, size, queue)
        if card is None:
            continue
        identity = (
            job_card_match_text(card["title"]),
            job_card_match_text(card["company"]),
        )
        if identity in existing_labels or (identity[0], "") in existing_labels:
            continue
        existing_labels.add(identity)
        queue.append(card)
    return queue, _job_results_memory(observation, collection, current_url)


def pending_job_cards(queue: list[dict]) -> list[dict]:
    return [
        dict(item)
        for item in queue or []
        if isinstance(item, dict) and str(item.get("status") or "pending") == "pending"
    ]


def resolved_job_card_count(queue: list[dict]) -> int:
    """이번 실행에서 수집했거나 DB 중복으로 확인한 카드 수를 반환한다."""

    return sum(
        1
        for item in queue or []
        if isinstance(item, dict)
        and (
            str(item.get("status") or "") == "done"
            or (
                str(item.get("status") or "") == "skipped"
                and str(item.get("job_id") or "").isdigit()
                and int(item["job_id"]) > 0
            )
        )
    )


def job_card_queue_scope_complete(
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
    return target_count > 0 and resolved_job_card_count(queue) >= target_count


def same_job_card(item: dict, args: dict) -> bool:
    if args.get("queue_id") and str(args.get("queue_id")) == str(item.get("queue_id")):
        return True
    label = str(args.get("target_label") or "").strip()
    if label and label == str(item.get("title") or "").strip():
        return True
    marker_id = args.get("marker_id")
    return marker_id is not None and str(marker_id) == str(item.get("source_marker_id"))


def active_job_card(queue: list[dict]) -> dict:
    """큐에서 현재 상세 화면과 연결된 활성 카드 하나를 반환한다."""

    return next(
        (
            dict(item)
            for item in queue or []
            if isinstance(item, dict) and item.get("status") == "active"
        ),
        {},
    )


def job_detail_key_from_state(state: WorkerState) -> str:
    """활성 카드 기준으로 상세 OCR 버퍼의 식별자를 만든다."""

    card = active_job_card(list(state["collection"].get("job_card_queue", []) or []))
    queue_id = str(card.get("queue_id") or "").strip()
    if queue_id:
        return queue_id
    company = str(card.get("company") or "").strip()
    title = str(card.get("title") or "").strip()
    return "|".join(part for part in (company, title) if part)


def activate_job_card(queue: list[dict], args: dict) -> list[dict]:
    updated = []
    activated = False
    for raw in queue or []:
        item = dict(raw)
        if (
            not activated
            and item.get("status") == "pending"
            and same_job_card(item, args)
        ):
            item["status"] = "active"
            activated = True
        updated.append(item)
    return updated


def job_card_click_matches_queue(queue: list[dict], args: dict) -> bool:
    if not queue:
        return False
    if args.get("queue_id"):
        return True
    component = str(args.get("target_component") or "")
    role = str(args.get("target_role") or "")
    if component in {"job_card", "job_card_title"} or role in {
        "job_card",
        "job_card_title",
    }:
        return True
    label = str(args.get("target_label") or "").strip()
    return bool(label) and any(
        label == str(item.get("title") or "").strip()
        for item in queue
        if isinstance(item, dict)
    )


def complete_active_job_card(queue: list[dict]) -> list[dict]:
    active_card = active_job_card(queue)
    if not active_card:
        return queue
    active_id = str(active_card.get("queue_id") or "")
    updated = []
    for raw in queue or []:
        item = dict(raw)
        if active_id and str(item.get("queue_id") or "") == active_id:
            item["status"] = "done"
        updated.append(item)
    return updated


def skip_active_job_card(
    queue: list[dict],
    *,
    reason: str,
    url: str,
    job_id: int | None = None,
) -> list[dict]:
    """이미 수집한 상세 URL의 활성 카드를 DB 근거가 확인된 상태로 끝낸다."""

    active_card = active_job_card(queue)
    if not active_card:
        return queue
    active_id = str(active_card.get("queue_id") or "")
    updated = []
    for raw in queue or []:
        item = dict(raw)
        if active_id and str(item.get("queue_id") or "") == active_id:
            item.update({"status": "skipped", "skip_reason": reason, "detail_url": url})
            if job_id is not None:
                item["job_id"] = int(job_id)
        updated.append(item)
    return updated


def job_results_page_matches(
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
    distance = hamming_distance(saved_phash, current_phash)
    overlap = (
        anchor_overlap(
            saved_signature.get("anchors") or [],
            (current_signature or {}).get("anchors") or [],
        )
        if require_anchors
        else None
    )

    settings = get_settings().reflex
    max_distance = settings.job_card_return_phash_max_distance
    min_overlap = settings.job_card_return_min_anchor_overlap
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


def normalized_return_action(value: Any) -> dict[str, Any]:
    """목록 복귀에 성공한 원자 행동만 재사용 가능한 형태로 정규화한다."""

    if not isinstance(value, dict):
        return {}
    name = str(value.get("name") or value.get("action") or "").strip()
    if name not in {"go_back", "close_current_tab", "switch_tab"}:
        return {}
    args = value.get("args") if isinstance(value.get("args"), dict) else {}
    if name == "switch_tab":
        direction = str(args.get("direction") or "").strip()
        if direction not in {"next", "previous"}:
            return {}
        return {"name": name, "args": {"direction": direction}}
    return {"name": name, "args": {}}


def return_action_from_transition(
    transition_result: dict,
) -> dict[str, Any]:
    step = (
        transition_result.get("step")
        if isinstance(transition_result.get("step"), dict)
        else {}
    )
    args = step.get("args") if isinstance(step.get("args"), dict) else {}
    return normalized_return_action(
        {
            "name": transition_result.get("action"),
            "args": args,
        }
    )


def job_card_marker_for_item(
    item: dict, markers: list[dict], signature: dict
) -> tuple[int | None, list[dict], dict]:
    target = dict(item.get("target") or {})
    target.setdefault("text", item.get("title", ""))
    target.setdefault("semantic_label", item.get("title", ""))
    target.setdefault("bbox_ratio", item.get("bbox_ratio") or [])
    target.setdefault("center_ratio", item.get("center_ratio") or [])
    target.setdefault("evidence_texts", item.get("evidence_texts") or [])
    marker_id = match_target_by_ratio(
        target,
        markers,
        screen_size_from_signature(signature),
    )
    if marker_id is not None:
        return marker_id, markers, {"reason": "current_marker_ratio_match"}

    bbox = bbox_from_ratio(
        item.get("bbox_ratio") or [], screen_size_from_signature(signature)
    )
    if bbox == [0, 0, 0, 0]:
        return None, markers, {"reason": "cached_bbox_missing"}
    next_id = (
        max(
            [
                int(marker.get("id") or 0)
                for marker in markers or []
                if isinstance(marker, dict)
            ]
            + [-1]
        )
        + 1
    )
    synthetic = {
        "id": next_id,
        "bbox": bbox,
        "text": item.get("title") or "queued job card",
        "type": "queue_cached_card",
    }
    return (
        next_id,
        [*markers, synthetic],
        {
            "reason": "synthetic_marker_from_cached_bbox",
            "bbox": bbox,
        },
    )


def replay_job_card_after_return(
    state: WorkerState,
    transition_result: dict,
    current_url: str,
    markers: list[dict],
    screen_signature: dict,
    *,
    require_anchors: bool = True,
) -> tuple[ActionRequest | None, list[dict], dict]:
    """어떤 물리 행동으로 복귀했든 목록 화면이 확인되면 다음 카드를 준비한다."""

    if not str(transition_result.get("action") or ""):
        return None, markers, {"reason": "return_transition_missing"}
    queue = [
        dict(item)
        for item in (state["collection"].get("job_card_queue", []) or [])
        if isinstance(item, dict)
    ]
    pending = pending_job_cards(queue)
    if not pending:
        return None, markers, {"reason": "queue_empty"}
    active_card = active_job_card(queue)
    if active_card:
        return (
            None,
            markers,
            {
                "reason": "active_card_not_completed",
                "queue_id": active_card.get("queue_id", ""),
                "title": active_card.get("title", ""),
            },
        )

    matched, match_trace = job_results_page_matches(
        dict(state["collection"].get("job_results_memory", {}) or {}),
        current_url,
        screen_signature,
        require_anchors=require_anchors,
    )
    if not matched:
        return None, markers, match_trace

    item = pending[0]
    marker_id, next_markers, marker_trace = job_card_marker_for_item(
        item, markers, screen_signature
    )
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
    request = build_action_request(
        "job_card_queue",
        "cached next job card",
        [
            {
                "name": "click_marker",
                "args": {
                    key: value for key, value in args.items() if key != "queue_id"
                },
                "id": f"job_card_queue_{item.get('queue_id', 'next')}",
                "metadata": {"queue_id": item.get("queue_id", "")},
            }
        ],
    )
    return (
        request,
        next_markers,
        {
            "hit": True,
            "queue_id": item.get("queue_id", ""),
            "title": item.get("title", ""),
            "return_match": match_trace,
            "marker": marker_trace,
        },
    )


__all__ = [
    "active_job_card",
    "resolved_job_card_count",
    "complete_active_job_card",
    "activate_job_card",
    "normalize_job_card_queue",
    "normalized_return_action",
    "pending_job_cards",
    "job_card_queue_scope_complete",
    "job_detail_key_from_state",
    "job_card_label",
    "job_card_marker_for_item",
    "job_card_match_text",
    "replay_job_card_after_return",
    "job_results_page_matches",
    "return_action_from_transition",
    "job_card_click_matches_queue",
    "job_card_entries_from_args",
    "same_job_card",
    "skip_active_job_card",
]
