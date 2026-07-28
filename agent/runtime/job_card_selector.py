"""검색 결과 화면에서 수집할 채용공고 카드만 짧게 선택한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.graph.action_request import build_action_request
from agent.graph.state import GraphState
from agent.recipe.page_context import normalize_page_role
from agent.runtime.action_validation import text_input_target_rejection
from agent.runtime.job_collection import job_count
from agent.runtime.job_card_queue import (
    job_card_queue_enabled,
    completed_job_card_count,
    job_card_queue_scope_complete,
)
from agent.runtime.site_context import site_runtime_guidance
from agent.runtime.transition_runtime import latest_no_effect_transition
from agent.utils.image_utils import image_to_base64_jpeg
from agent.utils.logger import logger
from agent.utils.model_dump import dump_model


class VisibleJobCard(BaseModel):
    """현재 화면에서 실제로 보이는 공고 카드 하나."""

    marker_id: int
    title: str
    company: str = Field(
        "",
        description="공고 카드에서 제목과 인접해 별도 표시된 회사명. 제목 괄호의 직무 분야는 회사명이 아님",
    )


class JobCardSelection(BaseModel):
    """검색 결과 여부와 수집 순서대로 고른 카드 목록."""

    is_job_results_page: bool = False
    is_loading: bool = Field(
        False,
        description="공고 제목 대신 스켈레톤·자리표시자만 보여 결과가 아직 로딩 중인지 여부",
    )
    needs_refinement: bool = False
    refinement_reason: str = ""
    refinement_action: Literal["click", "type", "none"] = "none"
    refinement_marker_id: int | None = None
    refinement_label: str = ""
    refinement_text: str = ""
    available_job_count: int | None = Field(
        None,
        ge=0,
        description="화면의 검색 결과 탭이나 결과 요약이 명시한 전체 공고 개수",
    )
    count_evidence: str = Field("", description="전체 결과 개수를 판단한 화면의 짧은 문구")
    count_confidence: float = Field(0.0, ge=0.0, le=1.0)
    cards: list[VisibleJobCard] = Field(default_factory=list)


def job_card_selector_enabled() -> bool:
    return get_settings().reflex.job_card_selector_enabled


def _target_count(state: GraphState) -> int:
    try:
        return max(0, int((state.get("recipe_params") or {}).get("target_count") or 0))
    except (TypeError, ValueError):
        return 0


def _count_mode(state: GraphState) -> str:
    params = state.get("recipe_params") or {}
    raw = params.get("count_mode") or ""
    return str(getattr(raw, "value", raw)).strip().lower()


def _collected_count(state: GraphState) -> int:
    return job_count(state.get("extracted_jd") or {})


def should_select_job_cards(state: GraphState) -> bool:
    """아직 큐가 없는 검색 결과 화면에만 전용 판단을 적용한다."""

    target_count = _target_count(state)
    collected_count = _collected_count(state)
    queue = [item for item in (state.get("job_card_queue") or []) if isinstance(item, dict)]
    count_mode = _count_mode(state)
    if job_card_queue_scope_complete(
        queue,
        count_mode=count_mode,
        target_count=target_count,
    ):
        return False
    needs_visible_screen = count_mode == "visible_all" and not queue
    queue_exhausted = bool(queue) and not any(
        str(item.get("status") or "pending") in {"pending", "active"}
        for item in queue
    )
    return bool(
        job_card_selector_enabled()
        and job_card_queue_enabled()
        and not latest_no_effect_transition(state)
        and normalize_page_role(state.get("current_page_role")) == "search"
        and (target_count > collected_count or needs_visible_screen)
        and (not queue or queue_exhausted)
        and not state.get("active_job_card")
        and state.get("current_markers")
        and state.get("marked_image")
    )


def _selector_model_name() -> str:
    from agent.application.model_policy import lightweight_model_name

    return lightweight_model_name("VISION_JOB_CARD_SELECTOR_MODEL")


def _get_job_card_selector_model() -> Any:
    from agent.application.model_clients import get_structured_google_model

    return get_structured_google_model(
        _selector_model_name(),
        JobCardSelection,
        temperature=0.0,
    )


def prepare_job_card_selector_model() -> None:
    """첫 검색 결과 판단 전에 구조화 모델을 생성해 둔다."""

    _get_job_card_selector_model()


def _compact_markers(markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        text = str(marker.get("text") or "").strip()
        try:
            marker_id = int(marker.get("id"))
        except (TypeError, ValueError):
            continue
        marker_type = str(marker.get("type") or "text").lower()
        input_candidate = False
        if not text:
            bbox = marker.get("bbox")
            if marker_type != "icon" or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                width = float(bbox[2]) - float(bbox[0])
                height = float(bbox[3]) - float(bbox[1])
            except (TypeError, ValueError):
                continue
            input_candidate = height > 0 and width / height >= 2.0
            if not input_candidate:
                continue
        compact.append(
            {
                "id": marker_id,
                "type": marker_type,
                "text": text[:160],
                "input_candidate": input_candidate,
            }
        )
    return compact


def _selection_messages(state: GraphState, remaining_count: int) -> list[Any]:
    markers = _compact_markers(list(state.get("current_markers") or []))
    recipe_params = dict(state.get("recipe_params") or {})
    search_query = str(recipe_params.get("query") or recipe_params.get("keyword") or "").strip()
    visible_all = _count_mode(state) == "visible_all" and _target_count(state) == 0
    settings = get_settings().vision
    max_dim = settings.reasoning_image_max_dim
    quality = settings.reasoning_image_quality
    image = image_to_base64_jpeg(
        Path(str(state.get("marked_image") or "")),
        max_dim=max_dim,
        quality=quality,
        fast=True,
    )
    selection_scope = (
        "현재 첫 안정 검색 결과 화면에 실제로 보이는 직접 관련 공고 전체"
        if visible_all
        else f"남은 목표 {remaining_count}개"
    )
    refinement_rule = (
        "검색 결과에 인접 직무가 섞여 있고 현재 보이는 사이트 필터로 검색어의 직무나 기술을 더 정확히 좁힐 수 있으면, "
        "카드를 고르기 전에 필터 정제를 먼저 하십시오. "
        if visible_all
        else (
            f"직접 일치하는 공고가 남은 목표 {remaining_count}개보다 적으면 비슷한 직무로 개수를 채우지 마십시오. "
        )
    )
    instruction = (
        "현재 화면이 채용공고 검색 결과 목록인지 판단하고, 실제로 보이는 공고 중 수집할 카드를 고르십시오. "
        "공고 카드 자리에 회색 스켈레톤·자리표시자만 반복되고 실제 공고 제목이 아직 보이지 않으면 is_loading을 true로 "
        "설정하고, is_job_results_page와 needs_refinement를 false로 두며 cards를 비우십시오. 로딩 중에는 검색어가 틀렸다고 "
        "판단하거나 검색 조건을 바꾸지 마십시오. "
        "사용자 검색어가 직무를 나타내면 공고 제목의 직무 정체성이 직접 일치해야 합니다. 기술 스택이나 업무 일부의 "
        "일치는 직무가 일치한 공고 사이의 순위 판단에만 사용하고, 제목이 다른 직무를 나타내는 공고를 직접 일치로 "
        "간주하지 마십시오. 검색어가 기술 자체만을 요구한 경우에만 제목 또는 기술 표기의 직접 일치를 사용하십시오. "
        "직접 일치하는 공고가 충분하면 "
        "단지 관련 기술이라는 이유만으로 범위가 더 넓거나 다른 직무의 공고를 섞지 마십시오. "
        f"수집 범위는 {selection_scope}입니다. "
        f"{refinement_rule}"
        "그 경우 needs_refinement를 true로 하고 cards를 비운 뒤, 현재 화면의 필터를 적용하거나 더 탐색해야 하는 이유를 "
        "refinement_reason에 짧게 적으십시오. 필터 정제에 바로 사용할 수 있는 컨트롤이 현재 보이면 refinement_action과 "
        "마커 정보를 채우십시오. 필터 메뉴가 닫혀 있으면 관련 옵션을 여는 컨트롤을 click하고, 검색 입력창이 보이지만 "
        "직접 일치 옵션이 아직 없으면 검색어를 type하고, 직접 일치 옵션이 보이면 그 옵션을 click하십시오. 적용 버튼은 "
        "직접 일치 옵션이 선택된 상태가 화면에서 확인될 때만 click하십시오. allowed_markers에서 input_candidate가 true인 "
        "무문자 마커는 화면상 입력창일 때 type 대상으로 선택할 수 있습니다. 필터 패널 뒤에 보이는 공고 카드의 텍스트는 "
        "필터 옵션이 아닙니다. refinement_label은 OCR text가 있으면 그대로 사용하고, 무문자 입력 후보일 때만 화면에서 "
        "파악한 입력창 이름을 쓰십시오. 입력창에 검색어가 적힌 것만으로는 옵션이 "
        "선택된 것이 아닙니다. type일 때는 refinement_text에 입력할 검색어를 넣으십시오. 이미 적용된 필터나 단지 범위가 "
        "비슷한 필터는 다시 선택하지 마십시오. 적합한 마커가 없으면 refinement_action을 none으로 하고 "
        "refinement_marker_id를 비우십시오. "
        "cards에는 공고 제목 자체에 붙은 마커 ID만 사용하고 회사명, 보상금, 배지, 버튼, 필터의 마커를 넣지 마십시오. "
        "각 card의 company에는 같은 카드에서 제목과 인접해 별도로 표시된 회사명만 넣으십시오. "
        "'Data Engineer(AI데이터플랫폼)'처럼 제목 괄호 안의 직무 분야나 조직명은 회사명으로 분리하지 마십시오. "
        "excluded_cards에 있는 제목과 회사의 공고는 이미 방문했으므로 cards에 다시 넣지 마십시오. "
        "검색 결과 탭이나 결과 요약에 전체 공고 개수가 명시되어 있으면 available_job_count에 넣고, 그 판단에 사용한 "
        "화면 문구를 count_evidence에 그대로 적으십시오. 페이지 번호, 알림, 필터 선택 개수는 결과 개수로 해석하지 마십시오. "
        "의미가 명확하지 않으면 available_job_count를 비우고 count_confidence를 낮게 두십시오. "
        f"cards는 검색어 관련성이 높은 순서로 최대 {remaining_count}개만 반환하고, 관련성이 같을 때만 화면 위에서 아래 순서를 따르십시오. "
        "숨겨진 카드나 화면에 없는 정보는 추측하지 마십시오. 검색 결과 목록이 아니거나 확실한 공고 제목을 찾지 못하면 "
        "is_job_results_page를 false로 하고 cards를 비우십시오."
    )
    current_url = str(state.get("current_url") or "")
    site_guidance = site_runtime_guidance(
        current_url,
        str(state.get("current_page_role") or "search"),
    )
    if site_guidance:
        instruction += "\n\n" + site_guidance.strip()
    payload = {
        "search_query": search_query,
        "count_mode": _count_mode(state),
        "remaining_count": remaining_count,
        "current_url": current_url,
        "allowed_markers": markers,
        "excluded_cards": [
            {
                "title": str(item.get("title") or ""),
                "company": str(item.get("company") or ""),
            }
            for item in (state.get("job_card_queue") or [])
            if isinstance(item, dict)
        ],
    }
    return [
        SystemMessage(content=instruction),
        HumanMessage(
            content=[
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
            ]
        ),
    ]


def _validated_cards(
    selection: dict[str, Any],
    markers: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    marker_ids = {
        int(marker.get("id")): marker
        for marker in markers or []
        if isinstance(marker, dict) and str(marker.get("id", "")).lstrip("-").isdigit()
    }
    cards: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for raw in selection.get("cards") or []:
        if not isinstance(raw, dict):
            continue
        try:
            marker_id = int(raw.get("marker_id"))
        except (TypeError, ValueError):
            continue
        if marker_id not in marker_ids or marker_id in used_ids:
            continue
        title = str(raw.get("title") or marker_ids[marker_id].get("text") or "").strip()
        if not title:
            continue
        cards.append(
            {
                "marker_id": marker_id,
                "title": title,
                "company": str(raw.get("company") or "").strip(),
            }
        )
        used_ids.add(marker_id)
        if len(cards) >= limit:
            break
    return cards


def _validated_refinement_target(
    selection: dict[str, Any],
    markers: list[dict[str, Any]],
) -> dict[str, Any] | None:
    action = str(selection.get("refinement_action") or "none").strip().lower()
    if action not in {"click", "type"}:
        return None
    try:
        marker_id = int(selection.get("refinement_marker_id"))
    except (TypeError, ValueError):
        return None
    marker = next(
        (
            item
            for item in markers or []
            if isinstance(item, dict) and str(item.get("id", "")).lstrip("-").isdigit()
            and int(item["id"]) == marker_id
        ),
        None,
    )
    if not marker:
        return None
    marker_label = str(marker.get("text") or "").strip()
    selected_label = str(selection.get("refinement_label") or "").strip()
    if marker_label:
        normalized_marker = "".join(char.casefold() for char in marker_label if char.isalnum())
        normalized_selected = "".join(char.casefold() for char in selected_label if char.isalnum())
        if (
            normalized_selected
            and normalized_marker not in normalized_selected
            and normalized_selected not in normalized_marker
        ):
            logger.warning(
                "Rejected refinement target because model label disagrees with OCR marker",
                marker_id=marker_id,
                marker_label=marker_label,
                selected_label=selected_label,
            )
            return None
    elif action != "type" or text_input_target_rejection(markers, marker_id) is not None:
        return None
    text = str(selection.get("refinement_text") or "").strip()
    if action == "type" and not text:
        return None
    label = marker_label or selected_label or f"input marker {marker_id}"
    return {"action": action, "marker_id": marker_id, "label": label, "text": text}


def select_job_cards(state: GraphState) -> tuple[Any | None, dict[str, Any]]:
    """전용 VLM 결과를 카드 큐 저장과 첫 카드 클릭 요청으로 변환한다."""

    if not should_select_job_cards(state):
        return None, {"attempted": False, "reason": "selector_not_applicable"}

    target_count = _target_count(state)
    resolved_count = max(
        _collected_count(state),
        completed_job_card_count(list(state.get("job_card_queue") or [])),
    )
    remaining_count = (
        target_count - resolved_count
        if target_count > 0
        else len(state.get("current_markers") or [])
    )
    try:
        from agent.application.run_context import invoke_with_metrics

        raw = invoke_with_metrics(
            _get_job_card_selector_model(),
            _selection_messages(state, remaining_count),
            "job_card_selection",
            stream=True,
        )
        selection = dump_model(raw)
    except Exception as exc:
        logger.warning("Job card selector failed; falling back to general reasoning", error=str(exc))
        return None, {
            "attempted": True,
            "reason": "selector_failed",
            "error": str(exc)[:200],
        }

    availability: dict[str, Any] = {}
    try:
        available_count = int(selection.get("available_job_count"))
        count_confidence = float(selection.get("count_confidence") or 0.0)
    except (TypeError, ValueError):
        available_count = -1
        count_confidence = 0.0
    count_evidence = str(selection.get("count_evidence") or "").strip()[:160]
    if available_count >= 0 and count_confidence >= 0.8 and count_evidence:
        availability = {
            "available_job_count": available_count,
            "count_evidence": count_evidence,
            "count_confidence": count_confidence,
        }
    if selection.get("is_loading"):
        return None, {
            "attempted": True,
            "reason": "screen_loading",
            "model": _selector_model_name(),
        }
    if not selection.get("is_job_results_page"):
        return None, {"attempted": True, "reason": "not_result_list", **availability}
    if selection.get("needs_refinement"):
        refinement_target = _validated_refinement_target(
            selection,
            list(state.get("current_markers") or []),
        )
        refinement_reason = str(selection.get("refinement_reason") or "").strip()[:300]
        if refinement_target:
            action_name = "type_in_marker" if refinement_target["action"] == "type" else "click_marker"
            action_args = {
                "marker_id": refinement_target["marker_id"],
                "target_label": refinement_target["label"],
                "target_role": "input" if action_name == "type_in_marker" else "filter",
                "target_component": "job_results_filter_input" if action_name == "type_in_marker" else "job_results_filter",
                "page_role": "search",
                "risk_level": "safe_navigation",
                "needs_user_confirmation": False,
                "reason": refinement_reason or "검색 결과를 요청에 맞게 좁힙니다.",
                "expected_after": (
                    "필터 검색어와 일치하는 선택지가 표시된다."
                    if action_name == "type_in_marker"
                    else "필터 선택지가 열리거나 검색 결과가 요청에 맞게 좁혀진다."
                ),
            }
            if action_name == "type_in_marker":
                action_args.update(
                    {
                        "text": refinement_target["text"],
                        "slot_name": "job_results_filter_query",
                    }
                )
            request = build_action_request(
                "card_selector",
                f"refine results with {action_name} on {refinement_target['label']}",
                [
                    {
                        "name": action_name,
                        "args": action_args,
                        "id": "card_selector_refinement",
                    }
                ],
            )
            logger.info(
                "Job card selector prepared refinement action",
                action=action_name,
                marker_id=refinement_target["marker_id"],
                label=refinement_target["label"],
            )
            return request, {
                "attempted": True,
                "reason": "job_results_refinement_action",
                "refinement_reason": refinement_reason,
                "action": action_name,
                "marker_id": refinement_target["marker_id"],
                "label": refinement_target["label"],
                "model": _selector_model_name(),
                **availability,
            }
        return None, {
            "attempted": True,
            "reason": "job_results_refinement_needed",
            "refinement_reason": refinement_reason,
            **availability,
        }
    cards = _validated_cards(
        selection,
        list(state.get("current_markers") or []),
        remaining_count,
    )
    excluded = {
        (
            str(item.get("title") or "").strip().casefold(),
            str(item.get("company") or "").strip().casefold(),
        )
        for item in (state.get("job_card_queue") or [])
        if isinstance(item, dict)
    }
    cards = [
        card
        for card in cards
        if (
            str(card.get("title") or "").strip().casefold(),
            str(card.get("company") or "").strip().casefold(),
        ) not in excluded
        and (str(card.get("title") or "").strip().casefold(), "") not in excluded
    ]
    if availability and availability["available_job_count"] < len(cards):
        availability = {}
    if not cards:
        return None, {"attempted": True, "reason": "no_valid_card", **availability}

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


__all__ = [
    "JobCardSelection",
    "VisibleJobCard",
    "prepare_job_card_selector_model",
    "job_card_selector_enabled",
    "select_job_cards",
    "should_select_job_cards",
]
