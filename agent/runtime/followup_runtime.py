"""검증된 직전 행동 문맥에서 좌표 없는 후속 행동을 재사용한다."""

from __future__ import annotations

from typing import Any

from agent.graph.action_request import ActionRequest, build_action_request
from agent.graph.state import GraphState
from agent.recipe.page_context import normalize_page_role
from agent.recipe.phash_replay import screen_context_signature_match
from agent.recipe.store import RecipeStore
from agent.recipe.task_category import normalize_task_category
from agent.recipe.text_utils import url_template
from agent.runtime.site_context import site_profile_for_url
from agent.utils.model_dump import dump_model


_ACTION_PARAMETER_KEYS = {
    "press_key": {"key"},
    "go_back": set(),
    "close_current_tab": set(),
    "switch_tab": {"direction"},
}


def _site_slug(state: GraphState, current_url: str) -> str:
    profile = site_profile_for_url(current_url)
    if profile is not None:
        return str(profile.slug or "")
    params = dict(state.get("recipe_params", {}) or {})
    return str(params.get("site") or "")


def _task_category(state: GraphState) -> str:
    params = dict(state.get("recipe_params", {}) or {})
    return normalize_task_category(params.get("task_category"))


def select_followup_action(
    state: GraphState,
    *,
    trigger_action: str,
    trigger_component: str = "",
    trigger_page_role: str = "",
    page_role: str = "",
    current_url: str = "",
    db_path=None,
) -> tuple[ActionRequest | None, dict[str, Any]]:
    """현재 직전 행동과 정확히 맞는 후속 전략을 검증된 행동 요청으로 바꾼다."""

    resolved_url = str(current_url or state.get("current_url") or "")
    site = _site_slug(state, resolved_url)
    task_category = _task_category(state)
    if not site or not trigger_action:
        return None, {
            "hit": False,
            "reason": "followup_scope_missing",
        }

    match = RecipeStore(db_path).get_followup_strategy(
        site,
        task_category=task_category,
        trigger_action=trigger_action,
        trigger_component=str(trigger_component or ""),
        trigger_page_role=normalize_page_role(trigger_page_role),
        page_role=normalize_page_role(page_role),
        current_url_template=url_template(resolved_url),
    )
    if match is None:
        return None, {
            "hit": False,
            "reason": "followup_strategy_missing",
            "trigger_action": trigger_action,
            "trigger_component": str(trigger_component or ""),
        }

    strategy_key, strategy = match
    context_match = screen_context_signature_match(
        dict(strategy.screen_context_signature or {}),
        dict(state.get("screen_signature") or {}),
    )
    if not context_match.get("matched"):
        return None, {
            "hit": False,
            "reason": str(
                context_match.get("reason")
                or "screen_context_mismatch"
            ),
            "strategy_key": strategy_key,
            "screen_context": context_match,
        }
    allowed_keys = _ACTION_PARAMETER_KEYS.get(strategy.action)
    if allowed_keys is None:
        return None, {
            "hit": False,
            "reason": "followup_action_not_allowed",
            "strategy_key": strategy_key,
        }
    args = {
        key: value
        for key, value in dict(strategy.param or {}).items()
        if key in allowed_keys
    }
    args.update(
        {
            "reason": "자율 탐색에서 검증된 직전 행동의 후속 전략을 재사용합니다.",
            "expected_after": strategy.expected_after,
            "page_role": strategy.page_role or page_role or None,
            "risk_level": "safe_navigation",
            "needs_user_confirmation": False,
        }
    )
    trigger = dump_model(strategy.trigger)
    contract = dump_model(strategy.transition_contract)
    request = build_action_request(
        "followup_strategy",
        (
            f"reuse {strategy.action} after "
            f"{trigger_action}:{trigger_component or '*'}"
        ),
        [
            {
                "name": strategy.action,
                "args": args,
                "id": f"followup_{strategy_key}",
                "metadata": {
                    "strategy_key": strategy_key,
                    "trigger": trigger,
                    "transition_contract": contract,
                    "transition_source": "followup_strategy",
                },
            }
        ],
        metadata={
            "strategy_key": strategy_key,
            "trigger": trigger,
        },
    )
    return request, {
        "hit": True,
        "strategy_key": strategy_key,
        "site": site,
        "task_category": task_category,
        "trigger": trigger,
        "action": strategy.action,
        "screen_context": context_match,
    }


def select_followup_after_transition(
    state: GraphState,
    transition_result: dict[str, Any],
    *,
    db_path=None,
) -> tuple[ActionRequest | None, dict[str, Any]]:
    """준비 완료된 직전 화면 행동을 후속 전략의 트리거로 사용한다."""

    if str(transition_result.get("status") or "") != "ready":
        return None, {
            "hit": False,
            "reason": "trigger_transition_not_ready",
        }
    step = (
        transition_result.get("step")
        if isinstance(transition_result.get("step"), dict)
        else {}
    )
    args = step.get("args") if isinstance(step.get("args"), dict) else {}
    return select_followup_action(
        state,
        trigger_action=str(transition_result.get("action") or ""),
        trigger_component=str(
            step.get("component")
            or args.get("target_component")
            or ""
        ),
        trigger_page_role=str(
            step.get("page_role")
            or args.get("page_role")
            or ""
        ),
        # 직전 전환 계약이 준비 완료를 보장하므로 URL 기반 화면 역할 재추론을
        # 다시 필터로 쓰지 않는다. 오버레이는 같은 URL에서 home으로 보일 수 있다.
        page_role="",
        current_url=str(state.get("current_url") or ""),
        db_path=db_path,
    )


__all__ = [
    "select_followup_action",
    "select_followup_after_transition",
]
