"""검색 결과 화면에서 수집할 공고 카드만 짧게 선택한다."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.graph.action_request import build_action_message
from agent.graph.state import GraphState
from agent.recipe.page_context import normalize_page_role
from agent.runtime.result_card_queue import card_queue_enabled
from agent.utils.image_utils import image_to_base64_jpeg
from agent.utils.logger import logger
from agent.utils.model_dump import dump_model


class VisibleResultCard(BaseModel):
    """현재 화면에서 실제로 보이는 공고 카드 하나."""

    marker_id: int
    title: str
    company: str = ""


class ResultCardSelection(BaseModel):
    """검색 결과 여부와 수집 순서대로 고른 카드 목록."""

    is_result_list: bool = False
    cards: list[VisibleResultCard] = Field(default_factory=list)


def result_card_selector_enabled() -> bool:
    raw = os.getenv("VISION_RESULT_CARD_SELECTOR_ENABLED", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _target_count(state: GraphState) -> int:
    try:
        return max(0, int((state.get("recipe_params") or {}).get("target_count") or 0))
    except (TypeError, ValueError):
        return 0


def _collected_count(state: GraphState) -> int:
    extracted = state.get("extracted_jd") or {}
    if not isinstance(extracted, dict) or not extracted:
        return 0
    for value in extracted.values():
        if isinstance(value, list):
            return sum(1 for item in value if isinstance(item, dict) and item)
    return 1


def should_select_result_cards(state: GraphState) -> bool:
    """아직 큐가 없는 검색 결과 화면에만 전용 판단을 적용한다."""

    target_count = _target_count(state)
    return bool(
        result_card_selector_enabled()
        and card_queue_enabled()
        and normalize_page_role(state.get("current_page_role")) == "search"
        and target_count > _collected_count(state)
        and not state.get("result_card_queue")
        and not state.get("active_result_card")
        and state.get("current_markers")
        and state.get("marked_image")
    )


def _selector_model_name() -> str:
    return os.getenv("VISION_RESULT_CARD_SELECTOR_MODEL", "gemini-3.5-flash")


def _get_result_card_selector_model() -> Any:
    from agent.application.model_clients import get_structured_google_model

    return get_structured_google_model(
        _selector_model_name(),
        ResultCardSelection,
        temperature=0.0,
    )


def prepare_result_card_selector_model() -> None:
    """첫 검색 결과 판단 전에 구조화 모델을 생성해 둔다."""

    _get_result_card_selector_model()


def _compact_text_markers(markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        text = str(marker.get("text") or "").strip()
        if not text:
            continue
        try:
            marker_id = int(marker.get("id"))
        except (TypeError, ValueError):
            continue
        compact.append(
            {
                "id": marker_id,
                "type": str(marker.get("type") or "text"),
                "text": text[:160],
            }
        )
    return compact


def _selection_messages(state: GraphState, remaining_count: int) -> list[Any]:
    markers = _compact_text_markers(list(state.get("current_markers") or []))
    recipe_params = dict(state.get("recipe_params") or {})
    search_query = str(recipe_params.get("query") or recipe_params.get("keyword") or "").strip()
    try:
        max_dim = int(os.getenv("VISION_REASONING_IMAGE_MAX_DIM", "768"))
        quality = int(os.getenv("VISION_REASONING_IMAGE_QUALITY", "60"))
    except ValueError:
        max_dim = 768
        quality = 60
    image = image_to_base64_jpeg(
        Path(str(state.get("marked_image") or "")),
        max_dim=max_dim,
        quality=quality,
        fast=True,
    )
    instruction = (
        "현재 화면이 채용공고 검색 결과 목록인지 판단하고, 실제로 보이는 공고 중 수집할 카드를 고르십시오. "
        "사용자의 검색어와 직무명 또는 핵심 기술이 직접 일치하는 공고를 우선하십시오. 직접 일치하는 공고가 충분하면 "
        "단지 관련 기술이라는 이유만으로 범위가 더 넓거나 다른 직무의 공고를 섞지 마십시오. "
        "공고 제목 자체에 붙은 마커 ID를 사용하고 회사명, 보상금, 배지, 버튼, 필터의 마커는 선택하지 마십시오. "
        f"cards는 검색어 관련성이 높은 순서로 최대 {remaining_count}개만 반환하고, 관련성이 같을 때만 화면 위에서 아래 순서를 따르십시오. "
        "숨겨진 카드나 화면에 없는 정보는 추측하지 마십시오. 검색 결과 목록이 아니거나 확실한 공고 제목을 찾지 못하면 "
        "is_result_list를 false로 하고 cards를 비우십시오."
    )
    payload = {
        "search_query": search_query,
        "remaining_count": remaining_count,
        "current_url": str(state.get("current_url") or ""),
        "allowed_markers": markers,
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


def select_result_cards(state: GraphState) -> tuple[Any | None, dict[str, Any]]:
    """전용 VLM 결과를 카드 큐 저장과 첫 카드 클릭 요청으로 변환한다."""

    if not should_select_result_cards(state):
        return None, {"attempted": False, "reason": "selector_not_applicable"}

    remaining_count = _target_count(state) - _collected_count(state)
    try:
        from agent.application.run_context import invoke_with_metrics

        raw = invoke_with_metrics(
            _get_result_card_selector_model(),
            _selection_messages(state, remaining_count),
            "result_card_selection",
        )
        selection = dump_model(raw)
    except Exception as exc:
        logger.warning("Result card selector failed; falling back to general reasoning", error=str(exc))
        return None, {
            "attempted": True,
            "reason": "selector_failed",
            "error": str(exc)[:200],
        }

    if not selection.get("is_result_list"):
        return None, {"attempted": True, "reason": "not_result_list"}
    cards = _validated_cards(
        selection,
        list(state.get("current_markers") or []),
        remaining_count,
    )
    if not cards:
        return None, {"attempted": True, "reason": "no_valid_card"}

    first = cards[0]
    message = build_action_message(
        "card_selector",
        f"selected {len(cards)} visible result card(s)",
        [
            {
                "name": "set_result_card_queue",
                "args": {
                    "cards": cards,
                    "reason": "현재 검색 결과에서 수집할 공고를 작업 큐에 저장합니다.",
                },
                "id": "card_selector_queue",
            },
            {
                "name": "click_marker",
                "args": {
                    "marker_id": first["marker_id"],
                    "target_label": first["title"],
                    "target_role": "job_card",
                    "target_component": "job_card_title",
                    "page_role": "search",
                    "risk_level": "safe_navigation",
                    "needs_user_confirmation": False,
                    "reason": "작업 큐의 첫 번째 공고 상세 페이지를 엽니다.",
                    "expected_after": "선택한 공고의 상세 페이지가 열린다.",
                },
                "id": "card_selector_click",
            },
        ],
    )
    logger.info(
        "Result card selector prepared queue",
        card_count=len(cards),
        first_marker_id=first["marker_id"],
        first_title=first["title"],
    )
    return message, {
        "attempted": True,
        "reason": "cards_selected",
        "card_count": len(cards),
        "marker_ids": [card["marker_id"] for card in cards],
        "model": _selector_model_name(),
    }


__all__ = [
    "ResultCardSelection",
    "VisibleResultCard",
    "prepare_result_card_selector_model",
    "result_card_selector_enabled",
    "select_result_cards",
    "should_select_result_cards",
]
