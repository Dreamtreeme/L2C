"""채용공고 카드 큐를 만들고 확인된 목록 화면에서 다음 카드를 재생한다."""

from __future__ import annotations

from typing import Any

from agent.runtime.worker_contracts import (
    ActionRequest,
    ScreenMarker,
    WorkerState,
    build_action_request,
)
from agent.utils.text import normalize_text
from agent.runtime.site_context import looks_like_job_detail_url, normalize_page_role
from agent.runtime.worker_state import target_count_from_state
from agent.vision.marker_geometry import marker_bbox
from agent.vision.target_snapshot import marker_by_id


def job_card_label(card: dict) -> str:
    return str(card.get("title") or "").strip()


def job_card_entries_from_args(args: dict) -> list[dict]:
    """도구 입력을 카드 후보 목록으로 통일한다."""

    return [dict(card) for card in args.get("cards") or []]


def job_card_match_text(value: Any) -> str:
    text = normalize_text(value)
    return "".join(char for char in text.casefold() if char.isalnum())


def job_card_is_known(card: dict, known_cards: list[dict]) -> bool:
    """제목과 회사가 같은 카드를 이미 처리한 카드로 판정한다."""

    title = job_card_match_text(card.get("title"))
    company = job_card_match_text(card.get("company"))
    if not title:
        return False
    for known in known_cards:
        known_title = job_card_match_text(known.get("title"))
        known_company = job_card_match_text(known.get("company"))
        if title != known_title:
            continue
        if not company or not known_company or company == known_company:
            return True
    return False


def _normalized_job_card(
    raw: dict[str, Any],
    markers: list[ScreenMarker],
    queue: list[dict],
    observation_id: str,
    screenshot_path: str,
) -> dict[str, Any] | None:
    marker_id = int(raw["marker_id"])
    marker = marker_by_id(markers, marker_id)
    if marker is None:
        return None
    label = job_card_label(raw)
    if not label:
        return None
    company = str(raw.get("company") or "").strip()
    return {
        "queue_id": f"card-{len(queue) + 1}",
        "status": "pending",
        "title": label,
        "company": company,
        "source_marker_id": marker_id,
        "source_marker_text": str(marker.get("text") or "").strip(),
        "source_marker_bbox": marker_bbox(marker),
        "source_observation_id": observation_id,
        "source_screenshot_path": screenshot_path,
    }


def normalize_job_card_queue(
    args: dict,
    state: WorkerState,
) -> list[dict]:
    """LLM이 고른 현재 화면의 공고 카드를 제목 기반 큐로 정규화한다."""

    cards = job_card_entries_from_args(args)
    observation = state["observation"]
    collection = state["collection"]
    markers = list(observation.get("current_markers", []) or [])
    queue: list[dict] = [
        dict(item)
        for item in (collection.get("job_card_queue", []) or [])
        if isinstance(item, dict) and str(item.get("status") or "") != "pending"
    ]
    observation_id = str(observation.get("observation_id") or "")
    screenshot_path = str(observation.get("current_screenshot") or "")
    for raw in cards:
        card = _normalized_job_card(
            raw,
            markers,
            queue,
            observation_id,
            screenshot_path,
        )
        if card is None:
            continue
        if job_card_is_known(card, queue):
            continue
        queue.append(card)
    return queue


def pending_job_cards(queue: list[dict]) -> list[dict]:
    return [
        dict(item)
        for item in queue or []
        if isinstance(item, dict) and str(item.get("status") or "pending") == "pending"
    ]


def has_unresolved_job_card_queue(state: WorkerState) -> bool:
    """현재 큐에 아직 선택하거나 수집 중인 카드가 있는지 반환한다."""

    return any(
        str(item.get("status") or "pending") in {"pending", "active"}
        for item in state["collection"].get("job_card_queue", []) or []
        if isinstance(item, dict)
    )


def can_select_pending_job_card(state: WorkerState) -> bool:
    """현재 관찰에서 대기 중인 다음 카드를 안전하게 선택할 수 있는지 판정한다."""

    observation = state["observation"]
    queue = list(state["collection"].get("job_card_queue", []) or [])
    current_url = str(observation.get("current_url") or "")
    return bool(
        observation.get("ocr_complete")
        and normalize_page_role(observation.get("current_page_role")) == "search"
        and not looks_like_job_detail_url(current_url)
        and pending_job_cards(queue)
        and not active_job_card(queue)
    )


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


def needs_job_results_navigation(state: WorkerState) -> bool:
    """상세 처리가 끝났고 다음 공고를 위해 목록 화면이 필요한지 계산한다."""

    transition = dict(state["transition"].get("transition_result") or {})
    action = str(transition.get("action") or "")
    source = str(transition.get("source") or "")
    outcome = str(transition.get("outcome") or transition.get("reason") or "")
    observation = state["observation"]
    current_url = str(observation.get("current_url") or "")
    last_review = state["collection"].get("last_job_review")
    rejected_detail = bool(
        last_review
        and last_review.status.value in {"source_incomplete", "invalid_target"}
        and str(last_review.url).split("#", 1)[0]
        == current_url.split("#", 1)[0]
    )
    detail_completed = any(
        capture.url == current_url
        for capture in state["collection"].get("job_captures", [])
    ) or outcome == "existing_job_detail" or rejected_detail
    navigation_continuing = (
        source == "job_results_navigation"
        and action in {"go_back", "close_current_tab"}
    )
    if not detail_completed and not navigation_continuing:
        return False

    queue = [
        dict(item)
        for item in state["collection"].get("job_card_queue", []) or []
        if isinstance(item, dict)
    ]
    if queue and active_job_card(queue):
        return False
    target_count = target_count_from_state(state)
    resolved_count = max(
        len(state["collection"].get("job_captures", [])),
        resolved_job_card_count(queue),
    )
    needs_more_cards = bool(pending_job_cards(queue)) or (
        target_count > 0 and resolved_count < target_count
    )
    if not needs_more_cards:
        return False
    return str(
        observation.get("current_page_role") or ""
    ) == "job_detail" or looks_like_job_detail_url(
        current_url
    )


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
    return str(card.get("queue_id") or "").strip()


def activate_job_card(queue: list[dict], args: dict) -> list[dict]:
    queue_id = str(args.get("queue_id") or "").strip()
    if not queue_id:
        return queue
    updated = []
    activated = False
    for raw in queue or []:
        item = dict(raw)
        if (
            not activated
            and item.get("status") == "pending"
            and queue_id == str(item.get("queue_id") or "")
        ):
            item["status"] = "active"
            activated = True
        updated.append(item)
    return updated


def job_card_click_matches_queue(queue: list[dict], args: dict) -> bool:
    queue_id = str(args.get("queue_id") or "").strip()
    return bool(queue_id) and any(
        queue_id == str(item.get("queue_id") or "")
        for item in queue or []
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


def reject_active_job_card(
    queue: list[dict],
    *,
    reason: str,
    url: str,
) -> list[dict]:
    """검토를 통과하지 못한 현재 카드만 제외하고 다음 후보를 남긴다."""

    active_card = active_job_card(queue)
    if not active_card:
        return queue
    active_id = str(active_card.get("queue_id") or "")
    updated = []
    for raw in queue or []:
        item = dict(raw)
        if active_id and str(item.get("queue_id") or "") == active_id:
            item.update(
                {
                    "status": "rejected",
                    "rejection_reason": reason,
                    "detail_url": url,
                }
            )
        updated.append(item)
    return updated


def job_card_marker_for_item(
    item: dict,
    markers: list[ScreenMarker],
    *,
    observation_id: str = "",
) -> tuple[int | None, dict]:
    """현재 OCR에서 큐에 저장된 공고 제목과 가장 잘 맞는 마커를 찾는다."""

    if observation_id and observation_id == str(item.get("source_observation_id") or ""):
        source_marker = marker_by_id(markers, item.get("source_marker_id"))
        if source_marker is not None:
            return int(source_marker["id"]), {"reason": "source_observation_marker"}

    saved_text = job_card_match_text(
        item.get("source_marker_text") or item.get("title")
    )
    if not saved_text:
        return None, {"reason": "queued_card_title_missing"}

    candidates: list[tuple[int, int, int]] = []
    for marker in markers:
        current_text = job_card_match_text(marker.get("text"))
        if not current_text:
            continue
        exact = int(current_text == saved_text)
        if not exact and saved_text not in current_text and current_text not in saved_text:
            continue
        candidates.append(
            (
                exact,
                min(len(saved_text), len(current_text)),
                int(marker["id"]),
            )
        )
    if not candidates:
        return None, {"reason": "current_marker_identity_missing"}

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    best = candidates[0]
    if len(candidates) > 1 and best[:2] == candidates[1][:2]:
        return None, {"reason": "current_marker_identity_ambiguous"}
    return best[2], {"reason": "current_marker_identity_match"}


def next_job_card_request(
    state: WorkerState,
    markers: list[ScreenMarker],
) -> tuple[ActionRequest | None, dict]:
    """목록 화면과 현재 OCR이 확인되면 큐의 다음 카드 행동을 만든다."""

    if not can_select_pending_job_card(state):
        return None, {"reason": "pending_card_not_selectable"}
    queue = [
        dict(item)
        for item in (state["collection"].get("job_card_queue", []) or [])
        if isinstance(item, dict)
    ]
    pending = pending_job_cards(queue)
    if not pending:
        return None, {"reason": "queue_empty"}
    active_card = active_job_card(queue)
    if active_card:
        return (
            None,
            {
                "reason": "active_card_not_completed",
                "queue_id": active_card.get("queue_id", ""),
                "title": active_card.get("title", ""),
            },
        )

    item = pending[0]
    marker_id, marker_trace = job_card_marker_for_item(
        item,
        markers,
        observation_id=str(state["observation"].get("observation_id") or ""),
    )
    if marker_id is None:
        return None, marker_trace
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
        "next queued job card",
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
        {
            "hit": True,
            "queue_id": item.get("queue_id", ""),
            "title": item.get("title", ""),
            "marker": marker_trace,
        },
    )


__all__ = [
    "active_job_card",
    "resolved_job_card_count",
    "complete_active_job_card",
    "activate_job_card",
    "normalize_job_card_queue",
    "needs_job_results_navigation",
    "pending_job_cards",
    "job_card_queue_scope_complete",
    "can_select_pending_job_card",
    "has_unresolved_job_card_queue",
    "job_detail_key_from_state",
    "job_card_is_known",
    "job_card_label",
    "job_card_marker_for_item",
    "job_card_match_text",
    "next_job_card_request",
    "reject_active_job_card",
    "job_card_click_matches_queue",
    "job_card_entries_from_args",
    "skip_active_job_card",
]
