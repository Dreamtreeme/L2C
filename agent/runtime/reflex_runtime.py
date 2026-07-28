"""ROI 서명이 일치하는 Reflex Recipe를 구조화된 행동 요청으로 재생한다."""

from __future__ import annotations

import time
from typing import Any

from agent.graph.action_request import build_action_request
from agent.graph.state import GraphState
from agent.runtime.action_validation import text_input_target_rejection
from agent.runtime.transition_runtime import used_idempotent_recipe_keys_on_url
from agent.utils.logger import logger
from agent.utils.model_dump import dump_model


def _reflex_trace_args(step: dict) -> dict:
    """실행에 영향 없는 레시피 추적 메타데이터를 도구 인자로 복원한다."""

    out: dict[str, str] = {}
    mapping = {
        "intent": "reason",
        "page_role": "page_role",
        "target_role": "target_role",
        "component": "target_component",
        "expected_after": "expected_after",
    }
    for source_key, arg_key in mapping.items():
        value = step.get(source_key)
        if value:
            out[arg_key] = str(value)
    target = step.get("target") if isinstance(step.get("target"), dict) else {}
    semantic_label = target.get("semantic_label")
    if semantic_label:
        out["target_label"] = str(semantic_label)
    return out


def _reflex_action_args(step: dict, marker_id: int | None, params: dict | None = None) -> dict | None:
    """저장된 RecipeStep을 action 도구 인자로 변환한다."""

    action = step.get("action")
    param = dict(step.get("param") or {})
    value = step.get("value")
    params = dict(params or {})
    trace_args = _reflex_trace_args(step)

    if action == "click_marker":
        return {"marker_id": marker_id, **trace_args} if marker_id is not None else None
    if action == "type_in_marker":
        if marker_id is None:
            return None
        slot_name = param.get("slot_name") or param.get("slot") or ""
        if not slot_name:
            slot_refs = step.get("slot_refs") or []
            slot_name = slot_refs[0] if slot_refs else ""
        if step.get("replay_mode") == "parameterized" and not slot_name:
            return None
        text = params.get(slot_name) if slot_name else None
        if slot_name and step.get("replay_mode") == "parameterized" and not text:
            return None
        text = text or param.get("text") or value
        if not text:
            return None
        args = {"marker_id": marker_id, "text": text, **trace_args}
        if slot_name:
            args["slot_name"] = slot_name
        return args
    return None


def _missing_required_recipe_inputs(recipe: Any, params: dict) -> list[str]:
    """레시피 skill metadata의 필수 입력 중 누락된 이름을 반환한다."""

    metadata = getattr(recipe, "skill_metadata", None)
    inputs = getattr(metadata, "inputs", []) if metadata is not None else []
    missing: list[str] = []
    for item in inputs or []:
        name = getattr(item, "name", "")
        required = bool(getattr(item, "required", False))
        value = params.get(name)
        if required and name and (value is None or value == ""):
            missing.append(name)
    return missing


def attempt_reflex_replay(state: GraphState) -> dict[str, Any]:
    """일치하는 ROI Recipe가 있으면 reasoning을 우회해 행동 요청을 만든다."""

    started = time.perf_counter()
    logger.info("Executing Reflex Node")

    def miss(_elapsed: float, reason: str = "", trace: dict | None = None) -> dict[str, Any]:
        reflex_trace = dict(trace or {})
        reflex_trace.update({"hit": False, "reason": reason or reflex_trace.get("reason", "")})
        active_set = dict(state.get("reflex_action_set", {}) or {})
        blocked_keys = [
            str(key)
            for key in (state.get("reflex_blocked_recipe_keys") or [])
            if str(key)
        ]
        active_recipe_key = str(active_set.get("recipe_key") or "")
        if active_recipe_key and active_recipe_key not in blocked_keys:
            blocked_keys.append(active_recipe_key)
        return {
            "reflex_trace": reflex_trace,
            "reflex_transition_contracts": {},
            "reflex_action_set": {},
            "reflex_blocked_recipe_keys": blocked_keys,
        }

    try:
        from agent.recipe.matcher import is_replayable_step
        from agent.recipe.page_context import normalize_page_role, page_role_matches
        from agent.recipe.phash_replay import match_step_by_screen_signature
        from agent.recipe.store import RecipeStore
        from agent.recipe.text_utils import recipe_url_scope_matches, url_template

        markers = state.get("current_markers", []) or []
        params = dict(state.get("recipe_params") or {})
        params.setdefault("goal", state.get("goal", ""))
        requested_task_category = str(params.get("task_category") or "").strip()
        site = str(params.get("site") or "").strip()
        recent_images = state.get("recent_images", []) or []
        current_image_path = str(recent_images[-1]) if recent_images else ""
        current_page_role = normalize_page_role(state.get("current_page_role", ""))
        current_url = str(state.get("current_url") or "")
        current_url_template = url_template(current_url)
        transition_failed_recipe_keys = {
            str(key)
            for key in (state.get("reflex_blocked_recipe_keys") or [])
            if str(key)
        }
        already_used_recipe_keys = used_idempotent_recipe_keys_on_url(
            state,
            str(state.get("current_url") or ""),
        )
        blocked_recipe_keys = transition_failed_recipe_keys | already_used_recipe_keys
        recipe_candidates = (
            RecipeStore().get_site_recipes(
                site,
                task_category=requested_task_category or None,
            )
            if site
            else []
        )
        active_set = dict(state.get("reflex_action_set", {}) or {})
        active_recipe_key = str(active_set.get("recipe_key") or "")
        if active_recipe_key:
            recipe_candidates = [
                item
                for item in recipe_candidates
                if str(item[0]) == active_recipe_key
            ]

        if not recipe_candidates:
            elapsed = time.perf_counter() - started
            logger.info("Reflex miss: no recipe", site=site, task_category=requested_task_category)
            return miss(
                elapsed,
                "no_recipe",
                {
                    "candidate_count": 0,
                    "site": site,
                    "task_category": requested_task_category,
                },
            )

        selected = None
        rejected_count = 0
        last_reject_reason = ""
        reject_reason_priority = {
            "capture_size_mismatch": 100,
            "roi_phash_distance": 90,
            "target_ratio_miss": 75,
            "url_scope_mismatch": 65,
            "page_role_mismatch": 40,
        }
        reject_reason_score = -1
        reject_reason_counts: dict[str, int] = {}
        candidate_rejections: list[dict[str, Any]] = []

        def record_rejection(recipe_key: str, reason: str, trace: dict[str, Any] | None = None) -> None:
            nonlocal last_reject_reason, reject_reason_score
            resolved = str(reason or "candidate_invalid")
            reject_reason_counts[resolved] = reject_reason_counts.get(resolved, 0) + 1
            score = reject_reason_priority.get(resolved, 50)
            if score > reject_reason_score:
                reject_reason_score = score
                last_reject_reason = resolved
            item = {"recipe_key": recipe_key, "reason": resolved}
            if trace:
                item.update(
                    {
                        "page_role": trace.get("page_role", ""),
                        "current_page_role": trace.get("current_page_role", ""),
                        "url_template": trace.get("url_template", ""),
                        "current_url_template": trace.get("current_url_template", ""),
                        "action": trace.get("action", ""),
                        "phash": dict(trace.get("phash") or {}),
                    }
                )
            candidate_rejections.append(item)

        for recipe_key, recipe in recipe_candidates:
            if recipe_key in blocked_recipe_keys:
                rejected_count += 1
                reason = (
                    "recipe_blocked_after_transition_failure"
                    if recipe_key in transition_failed_recipe_keys
                    else "recipe_already_used_on_page"
                )
                record_rejection(recipe_key, reason)
                continue
            if _missing_required_recipe_inputs(recipe, params):
                rejected_count += 1
                record_rejection(recipe_key, "missing_required_inputs")
                continue
            if not recipe.steps:
                rejected_count += 1
                record_rejection(recipe_key, "empty_recipe")
                continue

            step_count = len(recipe.steps)
            step_index = (
                int(active_set.get("next_step_index") or 0)
                if recipe_key == active_recipe_key
                else 0
            )
            if step_index < 0 or step_index >= step_count:
                rejected_count += 1
                record_rejection(
                    recipe_key,
                    "action_set_step_out_of_range",
                )
                continue

            tool_calls = []
            transition_contracts: dict[str, dict] = {}
            tool_call_traces: dict[str, dict[str, Any]] = {}
            candidate_valid = True
            for index, recipe_step in [
                (step_index, recipe.steps[step_index])
            ]:
                step = dump_model(recipe_step)
                action = step.get("action")
                step_trace: dict[str, Any] = {
                    "seq": step.get("seq"),
                    "step_index": index,
                    "action": action,
                    "page_role": step.get("page_role", ""),
                    "current_page_role": current_page_role,
                    "url_template": step.get("url_template", ""),
                    "current_url_template": current_url_template,
                    "replay_mode": step.get("replay_mode"),
                    "match_mode": "none",
                    "target_text": (
                        (step.get("target") or {}).get("text")
                        if isinstance(step.get("target"), dict)
                        else ""
                    ),
                }
                if not is_replayable_step(step):
                    record_rejection(recipe_key, "not_replayable", step_trace)
                    candidate_valid = False
                    break
                if not recipe_url_scope_matches(step.get("url_template", ""), current_url):
                    record_rejection(recipe_key, "url_scope_mismatch", step_trace)
                    candidate_valid = False
                    break
                if not page_role_matches(step.get("page_role", ""), current_page_role):
                    record_rejection(recipe_key, "page_role_mismatch", step_trace)
                    candidate_valid = False
                    break
                if action not in {"click_marker", "type_in_marker"}:
                    record_rejection(recipe_key, "non_roi_action", step_trace)
                    candidate_valid = False
                    break

                marker_id, phash_result = match_step_by_screen_signature(
                    step,
                    dict(state.get("screen_signature", {}) or {}),
                    markers,
                    current_image_path=current_image_path,
                )
                step_trace["phash"] = phash_result
                step_trace["match_mode"] = phash_result.get("mode") or "roi_phash"
                if marker_id is None:
                    record_rejection(
                        recipe_key,
                        phash_result.get("reason", "phash_check_failed"),
                        step_trace,
                    )
                    candidate_valid = False
                    break
                if action == "type_in_marker":
                    target_rejection = text_input_target_rejection(markers, marker_id)
                    if target_rejection:
                        record_rejection(
                            recipe_key,
                            str(target_rejection.get("reason") or "invalid_text_input_target"),
                            step_trace,
                        )
                        candidate_valid = False
                        break
                step_trace["marker_id"] = marker_id
                args = _reflex_action_args(step, marker_id, params=params)
                if args is None:
                    record_rejection(recipe_key, "args_build_failed", step_trace)
                    candidate_valid = False
                    break

                call_id = f"reflex_{abs(hash(recipe_key))}_{index}"
                tool_calls.append({"name": action, "args": args, "id": call_id})
                step_trace.update({"accepted": True, "tool_call_id": call_id})
                tool_call_traces[call_id] = dict(step_trace)
                contract = step.get("transition_contract")
                if contract:
                    transition_contracts[call_id] = dict(contract)
            if candidate_valid and tool_calls:
                selected = (
                    recipe_key,
                    recipe,
                    tool_calls,
                    transition_contracts,
                    tool_call_traces,
                )
                break
            rejected_count += 1
            if not last_reject_reason:
                record_rejection(recipe_key, "candidate_invalid")

        if selected is None:
            elapsed = time.perf_counter() - started
            logger.info(
                "Reflex miss: no candidate passed marker matching",
                candidates=len(recipe_candidates),
                last_reason=last_reject_reason,
                reject_reasons=reject_reason_counts,
                candidate_rejections=candidate_rejections[:12],
            )
            return miss(
                elapsed,
                "no_candidate_passed",
                {
                    "candidate_count": len(recipe_candidates),
                    "rejected_count": rejected_count,
                    "last_reason": last_reject_reason,
                    "reject_reasons": reject_reason_counts,
                    "candidate_rejections": candidate_rejections[:12],
                },
            )

        recipe_key, recipe, tool_calls, transition_contracts, tool_call_traces = selected
        selected_call = tool_calls[0]
        selected_trace = tool_call_traces[selected_call["id"]]
        step_index = int(selected_trace.get("step_index") or 0)
        step_count = len(recipe.steps)
        action_set_state = (
            {
                "recipe_key": recipe_key,
                "next_step_index": step_index + 1,
                "step_count": step_count,
                "actions": [
                    str(item.action)
                    for item in recipe.steps
                ],
            }
            if step_count > 1
            else {}
        )
        request = build_action_request(
            "reflex",
            (
                "cached action set step"
                if step_count > 1
                else "cached atomic action"
            ),
            tool_calls,
            metadata=(
                {
                    "execution_unit": "transition_action_set",
                    "recipe_key": recipe_key,
                    "step_index": step_index,
                    "step_count": step_count,
                }
                if step_count > 1
                else None
            ),
        )
        elapsed = time.perf_counter() - started
        logger.info(
            "Reflex hit",
            recipe_key=recipe_key[:24],
            actions=[call["name"] for call in tool_calls],
            action_set_step=(
                f"{step_index + 1}/{step_count}"
                if step_count > 1
                else ""
            ),
            transition_contracts=len(transition_contracts),
            when_to_use=getattr(
                getattr(recipe, "skill_metadata", None),
                "when_to_use",
                "",
            )[:80],
            duration=f"{elapsed:.3f}s",
        )
        reflex_trace = {
            "hit": True,
            "recipe_key": recipe_key,
            "candidate_count": len(recipe_candidates),
            "task_category": requested_task_category,
            "actions": [call["name"] for call in tool_calls],
            "tool_calls": tool_call_traces,
            "action_set_step_index": (
                step_index if step_count > 1 else None
            ),
            "action_set_step_count": (
                step_count if step_count > 1 else None
            ),
        }
        return {
            "pending_action": request,
            "reflex_trace": reflex_trace,
            "reflex_transition_contracts": transition_contracts,
            "reflex_action_set": action_set_state,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        logger.debug("reflex node skipped", error=str(exc))
        return miss(elapsed, "exception", {"error": str(exc)})


__all__ = ["attempt_reflex_replay"]
