"""
Phase 0: 비전 런의 (상태 -> 타깃) 텍스트 매핑 기록.
action_node에서 호출되며, 전부 예외 안전 -> 실패해도 실제 실행 흐름에 영향 0.
"""

from __future__ import annotations

from agent.recipe.matcher import marker_ordinal, marker_region
from agent.recipe.state_key import compute_state_key, normalize_text, site_of, url_template
from agent.utils.logger import logger

_TARGET_ACTIONS = {"click_marker", "type_in_marker"}
_RECORDED_ACTIONS = _TARGET_ACTIONS | {"scroll", "press_key", "go_back"}


def _squash(s) -> str:
    return normalize_text(s).lower().replace(" ", "")


def _marker(markers, marker_id):
    for m in markers or []:
        if isinstance(m, dict) and m.get("id") == marker_id:
            return m
    return None


def record_ui_step(recorded_steps, state, action_name, args, seq) -> None:
    """UI 액션 디스패치 직후 호출. recorded_steps에 in-place append (예외 안전)."""
    try:
        if action_name not in _RECORDED_ACTIONS:
            return
        markers = state.get("current_markers", []) or []
        url = state.get("current_url", "") or ""
        goal = state.get("goal", "") or ""
        step = {
            "seq": seq,
            "state_key": compute_state_key(url, markers),
            "url_template": url_template(url),
            "action": action_name,
            "target": None,
            "value": None,
            "param": {},
            "is_param": False,
            "expected_next_state": None,
        }
        if action_name in _TARGET_ACTIONS:
            marker = _marker(markers, args.get("marker_id"))
            if not marker:
                return
            step["target"] = {
                "text": normalize_text(marker.get("text")),
                "region": marker_region(marker, markers),
                "ordinal": marker_ordinal(marker, markers),
            }
            if action_name == "type_in_marker":
                val = (args.get("text") or "").strip()
                step["value"] = val
                step["param"] = {"text": val}
                step["is_param"] = bool(val) and _squash(val) in _squash(goal)
        elif action_name == "scroll":
            step["value"] = args.get("direction", "down")
            step["param"] = {"direction": step["value"]}
        elif action_name == "press_key":
            step["value"] = args.get("key")
            step["param"] = {"key": step["value"]}
        recorded_steps.append(step)
    except Exception as e:
        logger.debug("reflex record_ui_step skipped", error=str(e))


def commit_if_finished(recorded_steps, state, current_url) -> None:
    """finish_task로 런 종료 시 recorded_steps를 SiteRecipe로 검증·저장 (예외 안전)."""
    try:
        steps = list(recorded_steps or [])
        if not steps:
            return
        for i, step in enumerate(steps):
            for later in steps[i + 1:]:
                if later.get("state_key") != step.get("state_key"):
                    step["expected_next_state"] = later.get("state_key")
                    break
        site = site_of(current_url) or (steps[0].get("url_template", "").split("/")[0]) or "unknown"
        goal = state.get("goal", "") or ""

        # pydantic 검증을 거쳐 정형화한 뒤 저장
        from shared.schema.recipe_schema import RecipeStep, SiteRecipe
        from agent.recipe.store import RecipeStore

        recipe = SiteRecipe(site=site, goal=goal, steps=[RecipeStep(**s) for s in steps])
        dumped_steps = [
            s.model_dump() if hasattr(s, "model_dump") else s.dict()
            for s in recipe.steps
        ]
        saved = RecipeStore().commit_recipe(recipe.site, recipe.goal, dumped_steps)
        logger.info("reflex recipe recorded", site=site, steps=len(steps), saved_states=saved)
    except Exception as e:
        logger.debug("reflex commit_if_finished skipped", error=str(e))
