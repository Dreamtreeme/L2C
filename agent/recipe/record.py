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


def _bbox(marker: dict) -> list[int]:
    raw = marker.get("bbox") or [0, 0, 0, 0]
    if not isinstance(raw, list) or len(raw) != 4:
        return [0, 0, 0, 0]
    return [int(v or 0) for v in raw]


def _center(marker: dict) -> tuple[int, int]:
    x1, y1, x2, y2 = _bbox(marker)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _has_letter(text: str) -> bool:
    return any(ch.isalpha() for ch in text or "")


def _text_counts(markers: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for marker in markers or []:
        if not isinstance(marker, dict):
            continue
        text = normalize_text(marker.get("text"))
        key = text.lower().replace(" ", "")
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts




def _collect_evidence_candidates(
    target_marker: dict,
    markers: list[dict],
    counts: dict[str, int],
    target_text: str,
    unique_only: bool,
    max_dx: int,
    max_dy: int,
) -> list[tuple[int, int, int, int, str]]:
    seen = {target_text} if target_text else set()
    tx, ty = _center(target_marker)
    scored = []
    for marker in markers or []:
        if not isinstance(marker, dict) or marker.get("id") == target_marker.get("id"):
            continue
        text = normalize_text(marker.get("text"))
        key = text.lower().replace(" ", "")
        if not text or text in seen or len(key) < 2 or not _has_letter(text):
            continue
        if unique_only and counts.get(key, 0) != 1:
            continue
        x, y = _center(marker)
        dx = abs(x - tx)
        dy = abs(y - ty)
        if dx > max_dx or dy > max_dy:
            continue
        seen.add(text)
        length_bonus = min(len(key), 40)
        rank = dy * 2 + dx - length_bonus * 4
        scored.append((rank, dy, dx, marker.get("id", 0), text))
    return sorted(scored)


def _evidence_texts_for_marker(
    target_marker: dict,
    markers: list[dict],
    max_items: int = 6,
    max_dx: int = 850,
    max_dy: int = 320,
) -> list[str]:
    target_text = normalize_text(target_marker.get("text"))
    counts = _text_counts(markers)
    scored = _collect_evidence_candidates(
        target_marker,
        markers,
        counts,
        target_text,
        unique_only=True,
        max_dx=max_dx,
        max_dy=max_dy,
    )
    if not scored:
        scored = _collect_evidence_candidates(
            target_marker,
            markers,
            counts,
            target_text,
            unique_only=False,
            max_dx=max_dx,
            max_dy=max_dy,
        )
    return [item[-1] for item in scored[:max_items]]
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
            target = {
                "text": normalize_text(marker.get("text")),
                "region": marker_region(marker, markers),
                "ordinal": marker_ordinal(marker, markers),
            }
            target_label = normalize_text(args.get("target_label") or args.get("semantic_label"))
            if target_label:
                target["semantic_label"] = target_label
            evidence_texts = _evidence_texts_for_marker(marker, markers)
            if evidence_texts:
                target["evidence_texts"] = evidence_texts
            step["target"] = target
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
