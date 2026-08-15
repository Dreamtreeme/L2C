"""검색 결과 화면에서 수집할 채용공고 카드만 짧게 선택한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from agent.config import get_settings
from agent.observability.run_context import invoke_with_metrics
from agent.runtime.worker_contracts import (
    ActionRequest,
    ScreenMarker,
    WorkerState,
    build_action_request,
)
from agent.runtime.tool_schema import VisibleJobCard
from agent.runtime.site_context import normalize_page_role, site_runtime_guidance
from agent.runtime.job_card_queue import (
    has_unresolved_job_card_queue,
    job_card_is_known,
    resolved_job_card_count,
)
from agent.runtime.worker_state import (
    job_capture_count,
    count_mode_from_state,
    target_count_from_state,
)
from agent.runtime.transition_runtime import latest_no_effect_transition
from agent.utils.image_utils import image_to_base64_jpeg
from agent.utils.logger import logger
from agent.utils.model_conversion import parse_model_payload


class JobCardSelection(BaseModel):
    """검색 결과 여부와 수집 순서대로 고른 카드 목록."""

    model_config = ConfigDict(extra="forbid")

    is_job_results_page: bool = False
    available_job_count: int | None = Field(
        None,
        ge=0,
        description="화면의 검색 결과 탭이나 결과 요약이 명시한 전체 공고 개수",
    )
    count_evidence: str = Field(
        "", description="전체 결과 개수를 판단한 화면의 짧은 문구"
    )
    cards: list[VisibleJobCard] = Field(default_factory=list)


def should_select_job_cards(state: WorkerState) -> bool:
    """목표가 남은 검색 결과 화면에서 새 카드 후보를 선택한다."""

    if has_unresolved_job_card_queue(state):
        return False

    target_count = target_count_from_state(state)
    collected_count = job_capture_count(state)
    count_mode = count_mode_from_state(state)
    needs_visible_screen = count_mode == "visible_all"
    return bool(
        not latest_no_effect_transition(state)
        and normalize_page_role(state["observation"].get("current_page_role"))
        == "search"
        and (target_count > collected_count or needs_visible_screen)
        and state["observation"].get("current_markers")
        and state["observation"].get("marked_image")
    )


def _selector_model_name() -> str:
    from agent.llm.policy import lightweight_model_name

    return get_settings().models.job_card_selector_model or lightweight_model_name()


def _get_job_card_selector_model() -> Any:
    from agent.llm.clients import get_structured_google_model

    return get_structured_google_model(
        _selector_model_name(),
        JobCardSelection,
        temperature=0.0,
        execution_role="lightweight",
    )


def prepare_job_card_selector_model() -> None:
    """첫 검색 결과 판단 전에 구조화 모델을 생성해 둔다."""

    _get_job_card_selector_model()


def _compact_markers(markers: list[ScreenMarker]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        text = str(marker.get("text") or "").strip()
        raw_marker_id = marker.get("id")
        if raw_marker_id is None:
            continue
        try:
            marker_id = int(raw_marker_id)
        except (TypeError, ValueError):
            continue
        if not text:
            continue
        compact.append(
            {
                "id": marker_id,
                "type": str(marker.get("type") or "text").lower(),
                "text": text[:160],
            }
        )
    return compact


def _selection_messages(
    state: WorkerState,
    remaining_count: int | None,
) -> list[Any]:
    markers = _compact_markers(list(state["observation"].get("current_markers") or []))
    search_query = state["request"]["collection_intent"].search_keyword.strip()
    visible_all = (
        count_mode_from_state(state) == "visible_all"
        and target_count_from_state(state) == 0
    )
    settings = get_settings().vision
    max_dim = settings.reasoning_image_max_dim
    quality = settings.reasoning_image_quality
    image = image_to_base64_jpeg(
        Path(str(state["observation"].get("marked_image") or "")),
        max_dim=max_dim,
        quality=quality,
        fast=True,
    )
    selection_scope = (
        "현재 첫 안정 검색 결과 화면에 실제로 보이는 직접 관련 공고 전체"
        if visible_all
        else f"남은 목표 {remaining_count or 0}개"
    )
    instruction = (
        "현재 화면이 채용공고 검색 결과 목록인지 판단하고, 실제로 보이는 공고 중 수집할 카드를 고르십시오. "
        "사용자 검색어가 직무를 나타내면 공고 제목의 직무 정체성이 직접 일치해야 합니다. 기술 스택이나 업무 일부의 "
        "일치는 직무가 일치한 공고 사이의 순위 판단에만 사용하고, 제목이 다른 직무를 나타내는 공고를 직접 일치로 "
        "간주하지 마십시오. 검색어가 기술 자체만을 요구한 경우에만 제목 또는 기술 표기의 직접 일치를 사용하십시오. "
        "직접 일치하는 공고가 충분하면 단지 관련 기술이라는 이유만으로 범위가 더 넓거나 다른 직무의 공고를 섞지 마십시오. "
        f"수집 범위는 {selection_scope}입니다. "
        "cards에는 공고 제목 자체에 붙은 마커 ID만 사용하고 회사명, 보상금, 배지, 버튼, 필터의 마커를 넣지 마십시오. "
        "각 card의 company에는 같은 카드에서 제목과 인접해 별도로 표시된 회사명만 넣으십시오. "
        "'Data Engineer(AI데이터플랫폼)'처럼 제목 괄호 안의 직무 분야나 조직명은 회사명으로 분리하지 마십시오. "
        "검색 결과 탭이나 결과 요약에 전체 공고 개수가 명시되어 있으면 available_job_count에 넣고, 그 판단에 사용한 "
        "화면 문구를 count_evidence에 그대로 적으십시오. 페이지 번호, 알림, 필터 선택 개수는 결과 개수로 해석하지 마십시오. "
        "의미가 명확하지 않으면 available_job_count를 비우십시오. "
        + (
            f"cards는 검색어 관련성이 높은 순서로 최대 {remaining_count}개만 반환하고, "
            if remaining_count is not None
            else "cards에는 현재 화면에 보이는 직접 관련 공고를 모두 반환하고, "
        )
        + "관련성이 같을 때만 화면 위에서 아래 순서를 따르십시오. "
        "숨겨진 카드나 화면에 없는 정보는 추측하지 마십시오. 검색 결과 목록이 아니거나 확실한 공고 제목을 찾지 못하면 "
        "is_job_results_page를 false로 하고 cards를 비우십시오."
    )
    current_url = str(state["observation"].get("current_url") or "")
    site_guidance = site_runtime_guidance(
        current_url,
        str(state["observation"].get("current_page_role") or "search"),
    )
    if site_guidance:
        instruction += "\n\n" + site_guidance.strip()
    payload = {
        "search_query": search_query,
        "count_mode": count_mode_from_state(state),
        "remaining_count": remaining_count,
        "current_url": current_url,
        "allowed_markers": markers,
        "processed_cards": [
            {
                "title": str(item.get("title") or ""),
                "company": str(item.get("company") or ""),
            }
            for item in state["collection"].get("job_card_queue", []) or []
            if isinstance(item, dict)
        ][-20:],
    }
    return [
        SystemMessage(content=instruction),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                },
            ]
        ),
    ]


def _marker_index(markers: list[ScreenMarker]) -> dict[int, ScreenMarker]:
    marker_ids: dict[int, ScreenMarker] = {}
    for marker in markers or []:
        raw_marker_id = marker.get("id")
        if raw_marker_id is None or not str(raw_marker_id).lstrip("-").isdigit():
            continue
        marker_ids[int(raw_marker_id)] = marker
    return marker_ids


def _selected_card(
    raw: Any,
    marker_ids: dict[int, ScreenMarker],
) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or raw.get("marker_id") is None:
        return None
    try:
        marker_id = int(raw["marker_id"])
    except (TypeError, ValueError):
        return None
    marker = marker_ids.get(marker_id)
    if marker is None:
        return None
    title = str(raw.get("title") or marker.get("text") or "").strip()
    if not title:
        return None
    return {
        "marker_id": marker_id,
        "title": title,
        "company": str(raw.get("company") or "").strip(),
    }


def _validated_cards(
    selection: dict[str, Any],
    markers: list[ScreenMarker],
    limit: int | None,
    known_cards: list[dict],
) -> list[dict[str, Any]]:
    marker_ids = _marker_index(markers)
    cards: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for raw in selection.get("cards") or []:
        card = _selected_card(raw, marker_ids)
        if card is None or card["marker_id"] in used_ids:
            continue
        if job_card_is_known(card, known_cards):
            continue
        cards.append(card)
        used_ids.add(card["marker_id"])
        if limit is not None and len(cards) >= limit:
            break
    return cards


def _remaining_job_count(state: WorkerState) -> int | None:
    target_count = target_count_from_state(state)
    resolved_count = max(
        job_capture_count(state),
        resolved_job_card_count(list(state["collection"].get("job_card_queue") or [])),
    )
    return target_count - resolved_count if target_count > 0 else None


def _invoke_job_card_selector(
    state: WorkerState,
    remaining_count: int | None,
) -> dict[str, Any]:
    raw = invoke_with_metrics(
        _get_job_card_selector_model(),
        _selection_messages(state, remaining_count),
        "job_card_selection",
        stream=True,
    )
    return parse_model_payload(raw, JobCardSelection).model_dump(mode="json")


def _selection_availability(
    selection: dict[str, Any],
) -> dict[str, Any]:
    availability: dict[str, Any] = {}
    raw_available_count = selection.get("available_job_count")
    try:
        available_count = (
            int(raw_available_count) if raw_available_count is not None else -1
        )
    except (TypeError, ValueError):
        available_count = -1
    count_evidence = str(selection.get("count_evidence") or "").strip()[:160]
    if available_count >= 0 and count_evidence:
        availability = {
            "available_job_count": available_count,
            "count_evidence": count_evidence,
        }
    return availability


def _build_queue_selection(
    cards: list[dict[str, Any]],
    availability: dict[str, Any],
) -> tuple[ActionRequest, dict[str, Any]]:
    request = build_action_request(
        "card_selector",
        f"selected {len(cards)} visible job card(s)",
        [
            {
                "name": "set_job_card_queue",
                "args": {
                    "cards": cards,
                    **availability,
                    "reason": "현재 검색 결과에서 수집할 공고를 작업 큐에 저장합니다.",
                },
                "id": "card_selector_queue",
            },
        ],
    )
    logger.info(
        "Job card selector prepared queue",
        card_count=len(cards),
        first_marker_id=cards[0]["marker_id"],
        first_title=cards[0]["title"],
    )
    return request, {
        "attempted": True,
        "reason": "cards_selected",
        "card_count": len(cards),
        "marker_ids": [card["marker_id"] for card in cards],
        "model": _selector_model_name(),
        **availability,
    }


def _build_continue_selection(
    availability: dict[str, Any],
) -> tuple[ActionRequest, dict[str, Any]]:
    """현재 결과 화면에 관련 공고가 없으면 다음 화면 범위를 확인한다."""

    request = build_action_request(
        "card_selector",
        "현재 화면에 직접 관련된 공고가 없어 다음 결과를 확인합니다.",
        [
            {
                "name": "scroll",
                "args": {
                    "direction": "down",
                    "amount": "page",
                    "target_role": "job_results",
                    "target_component": "job_card_list",
                    "reason": "현재 화면에 직접 관련된 공고 카드가 없습니다.",
                    "expected_after": "다음 검색 결과 카드가 보입니다.",
                    "page_role": "search",
                    "risk_level": "safe_read",
                },
                "id": "card_selector_continue",
            },
        ],
    )
    logger.info("Job card selector continues to next visible results")
    return request, {
        "attempted": True,
        "reason": "no_valid_card_continue",
        "card_count": 0,
        "model": _selector_model_name(),
        **availability,
    }


def select_job_cards(
    state: WorkerState,
) -> tuple[ActionRequest | None, dict[str, Any]]:
    """전용 VLM 결과를 아직 처리하지 않은 카드 큐로 변환한다."""

    if not should_select_job_cards(state):
        return None, {"attempted": False, "reason": "selector_not_applicable"}

    remaining_count = _remaining_job_count(state)
    try:
        selection = _invoke_job_card_selector(state, remaining_count)
    except Exception as exc:
        logger.warning(
            "Job card selector failed; falling back to general reasoning",
            error=str(exc),
        )
        return None, {
            "attempted": True,
            "reason": "selector_failed",
            "error": str(exc)[:200],
        }

    availability = _selection_availability(selection)
    if not selection.get("is_job_results_page"):
        return None, {
            "attempted": True,
            "reason": "not_result_list",
            **availability,
        }

    cards = _validated_cards(
        selection,
        list(state["observation"].get("current_markers") or []),
        remaining_count,
        [
            dict(item)
            for item in state["collection"].get("job_card_queue", []) or []
            if isinstance(item, dict)
        ],
    )
    if availability and availability["available_job_count"] < len(cards):
        availability = {}
    if not cards:
        return _build_continue_selection(availability)
    return _build_queue_selection(cards, availability)


__all__ = [
    "JobCardSelection",
    "VisibleJobCard",
    "prepare_job_card_selector_model",
    "select_job_cards",
    "should_select_job_cards",
]
