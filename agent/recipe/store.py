"""반사 레시피 저장소(RecipeStore).

채용공고 DB(jobs DB)와 같은 SQLite 파일에 ROI 검증용 레시피를 저장한다.
활성 레시피 조회는 화면 상태 해시가 아니라 사이트/작업분류 후보 집합과 ROI 검증으로 한다.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.recipe.page_context import normalize_page_role
from agent.recipe.payload_sanitizer import strip_replay_runtime_fields
from agent.recipe.task_category import normalize_task_category, task_category_matches
from agent.utils.model_dump import dump_model
from shared.schema.recipe_schema import (
    FollowupActionStrategy,
    RecipeStep,
    SiteRecipe,
)
from shared.schema.skill_schema import RecipeSkillMetadata


_RECIPE_KEY_VERSION = 3
_RECIPE_KEY_PREFIX = "roi3#"
_FOLLOWUP_KEY_VERSION = 1
_FOLLOWUP_KEY_PREFIX = "followup1#"
_FOLLOWUP_ACTIONS = {
    "press_key",
    "go_back",
    "close_current_tab",
    "switch_tab",
}
_INITIALIZED_DB_PATHS: set[Path] = set()
_SCHEMA_LOCK = threading.RLock()


class RecipeStore:
    def __init__(self, db_path=None):
        if db_path is None:
            # 설정 지연 로드(lazy import)로 모듈 import 시점의 부작용을 줄인다.
            from shared.config import DB_PATH

            db_path = DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        cache_key = self.db_path.resolve()
        with _SCHEMA_LOCK:
            if cache_key not in _INITIALIZED_DB_PATHS:
                self._ensure_schema()
                _INITIALIZED_DB_PATHS.add(cache_key)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='recipes'"
            ).fetchone()
            if row:
                columns = {
                    item["name"]
                    for item in conn.execute("PRAGMA table_info(recipes)").fetchall()
                }
                required_columns = {
                    "recipe_key",
                    "site",
                    "goal",
                    "steps_json",
                    "metadata_json",
                    "success_count",
                    "created_at",
                    "updated_at",
                }
                if not required_columns.issubset(columns):
                    conn.execute("DROP TABLE recipes")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recipes (
                    recipe_key    TEXT PRIMARY KEY,
                    site          TEXT NOT NULL,
                    goal          TEXT,
                    steps_json    TEXT NOT NULL,
                    metadata_json TEXT,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_recipes_site ON recipes(site)")
            conn.execute(
                "DELETE FROM recipes WHERE recipe_key NOT LIKE ?",
                (f"{_RECIPE_KEY_PREFIX}%",),
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS followup_action_strategies (
                    strategy_key  TEXT PRIMARY KEY,
                    site          TEXT NOT NULL,
                    task_category TEXT,
                    strategy_json TEXT NOT NULL,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_followup_action_strategies_site "
                "ON followup_action_strategies(site)"
            )
            conn.execute(
                "DELETE FROM followup_action_strategies WHERE strategy_key NOT LIKE ?",
                (f"{_FOLLOWUP_KEY_PREFIX}%",),
            )

    @staticmethod
    def _step_has_required_replay_fields(step: dict[str, Any]) -> bool:
        action = str(step.get("action") or "")
        if action not in {"click_marker", "type_in_marker"}:
            return False
        return bool(normalize_page_role(step.get("page_role")) and step.get("roi_signature"))

    @staticmethod
    def _dump_json(payload: Any) -> str:
        return json.dumps(payload or {}, ensure_ascii=False)

    @staticmethod
    def _metadata_dict(metadata: dict[str, Any] | RecipeSkillMetadata | None) -> dict[str, Any]:
        return dump_model(metadata)

    @staticmethod
    def _metadata_task_category(metadata: dict[str, Any] | RecipeSkillMetadata | None) -> str:
        return normalize_task_category(RecipeStore._metadata_dict(metadata).get("task_category"))

    @staticmethod
    def _metadata_matches_task_category(
        metadata: dict[str, Any] | RecipeSkillMetadata | None,
        task_category: str | None,
    ) -> bool:
        return task_category_matches(task_category, RecipeStore._metadata_task_category(metadata))

    @staticmethod
    def _recipe_key_for_step(
        site: str,
        step: dict[str, Any],
        metadata: dict[str, Any] | RecipeSkillMetadata | None = None,
    ) -> str:
        """변하는 좌표·해시가 아니라 재생 의도로 활성 레시피 키를 만든다."""

        meta = RecipeStore._metadata_dict(metadata)
        target = step.get("target") if isinstance(step.get("target"), dict) else {}
        replay_mode = str(step.get("replay_mode") or "")
        payload = {
            "key_version": _RECIPE_KEY_VERSION,
            "site": site or "",
            "task_category": normalize_task_category(meta.get("task_category")),
            "page_role": normalize_page_role(step.get("page_role")),
            "url_template": step.get("url_template") or "",
            "action": step.get("action") or "",
            "component": step.get("component") or "",
            "target_role": step.get("target_role") or "",
            "slot_refs": (
                sorted(str(item) for item in step.get("slot_refs") or [])
                if replay_mode == "parameterized"
                else []
            ),
            "target_region": target.get("region") or "",
        }
        digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return f"{_RECIPE_KEY_PREFIX}{digest}"

    @staticmethod
    def _followup_key_for_strategy(
        strategy: dict[str, Any],
    ) -> str:
        """실행 중 변하는 값이 아닌 후속 행동 문맥으로 저장 키를 만든다."""

        trigger = (
            strategy.get("trigger")
            if isinstance(strategy.get("trigger"), dict)
            else {}
        )
        payload = {
            "key_version": _FOLLOWUP_KEY_VERSION,
            "site": str(strategy.get("site") or ""),
            "task_category": normalize_task_category(
                strategy.get("task_category")
            ),
            "trigger_action": str(trigger.get("action") or ""),
            "trigger_component": str(trigger.get("component") or ""),
            "trigger_page_role": normalize_page_role(
                trigger.get("page_role")
            ),
            "page_role": normalize_page_role(strategy.get("page_role")),
            "url_template": str(strategy.get("url_template") or ""),
            "action": str(strategy.get("action") or ""),
            "param": dict(strategy.get("param") or {}),
        }
        digest = hashlib.sha1(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        return f"{_FOLLOWUP_KEY_PREFIX}{digest}"

    @staticmethod
    def _validated_followup_strategy(
        strategy: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(strategy, dict):
            return None
        if str(strategy.get("action") or "") not in _FOLLOWUP_ACTIONS:
            return None
        trigger = (
            strategy.get("trigger")
            if isinstance(strategy.get("trigger"), dict)
            else {}
        )
        if not str(trigger.get("action") or ""):
            return None
        if not strategy.get("transition_contract"):
            return None
        try:
            return dump_model(FollowupActionStrategy(**strategy))
        except (TypeError, ValueError):
            return None

    def _upsert_recipe_steps(
        self,
        site: str,
        goal: str,
        step: dict,
        metadata: dict[str, Any] | RecipeSkillMetadata | None = None,
    ) -> bool:
        """같은 ROI 레시피 키의 원자 행동을 저장하거나 갱신한다."""

        if not isinstance(step, dict):
            return False
        replay_step = strip_replay_runtime_fields(step)
        if not self._step_has_required_replay_fields(replay_step):
            return False
        recipe_key = self._recipe_key_for_step(site, replay_step, metadata=metadata)

        now = datetime.now().isoformat(timespec="seconds")
        steps_payload = json.dumps([replay_step], ensure_ascii=False)
        metadata_payload = self._dump_json(self._metadata_dict(metadata))
        with self._conn() as conn:
            row = conn.execute("SELECT 1 FROM recipes WHERE recipe_key=?", (recipe_key,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE recipes "
                    "SET site=?, goal=?, steps_json=?, metadata_json=?, success_count=success_count+1, updated_at=? "
                    "WHERE recipe_key=?",
                    (site, goal, steps_payload, metadata_payload, now, recipe_key),
                )
            else:
                conn.execute(
                    "INSERT INTO recipes "
                    "(recipe_key, site, goal, steps_json, metadata_json, success_count, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,1,?,?)",
                    (recipe_key, site, goal, steps_payload, metadata_payload, now, now),
                )
        return True

    def commit_recipe(
        self,
        site: str,
        goal: str,
        steps: list[dict],
        metadata: dict[str, Any] | RecipeSkillMetadata | None = None,
    ) -> int:
        """성공 후보의 재생 가능한 행동을 ROI 레시피 단위로 저장한다."""

        saved = 0
        for step in steps or []:
            if not isinstance(step, dict):
                continue
            if self._upsert_recipe_steps(site, goal, dict(step), metadata=metadata):
                saved += 1
        return saved

    def clear_recipes(self, site: str | None = None) -> int:
        """활성 ROI 레시피와 후속 행동 전략만 비우고 후보 증거는 보존한다."""

        with self._conn() as conn:
            if site:
                recipe_result = conn.execute(
                    "DELETE FROM recipes WHERE site=?",
                    (site,),
                )
                followup_result = conn.execute(
                    "DELETE FROM followup_action_strategies WHERE site=?",
                    (site,),
                )
            else:
                recipe_result = conn.execute("DELETE FROM recipes")
                followup_result = conn.execute(
                    "DELETE FROM followup_action_strategies"
                )
            return int(recipe_result.rowcount or 0) + int(
                followup_result.rowcount or 0
            )

    def replace_recipe_steps(
        self,
        site: str,
        goal: str,
        source_steps: list[dict],
        replay_steps: list[dict],
        metadata: dict[str, Any] | RecipeSkillMetadata | None = None,
    ) -> int:
        """한 후보가 만든 기존 ROI 레시피를 지우고 승인된 재생 단계만 교체 저장한다."""
        metadata_dict = self._metadata_dict(metadata)
        recipe_keys = sorted(
            {
                self._recipe_key_for_step(site, step, metadata=metadata_dict)
                for step in [*(source_steps or []), *(replay_steps or [])]
                if isinstance(step, dict) and step.get("roi_signature")
            }
        )
        if recipe_keys:
            placeholders = ",".join("?" for _ in recipe_keys)
            with self._conn() as conn:
                conn.execute(
                    f"DELETE FROM recipes WHERE site=? AND recipe_key IN ({placeholders})",
                    (site, *recipe_keys),
                )
        return self.commit_recipe(site, goal, replay_steps, metadata=metadata)

    def get_by_site(self, site: str):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT recipe_key, site, goal, steps_json, metadata_json, success_count FROM recipes "
                "WHERE site=? ORDER BY success_count DESC",
                (site,),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["steps"] = json.loads(item.pop("steps_json") or "[]")
            item["skill_metadata"] = json.loads(item.pop("metadata_json") or "{}")
            out.append(item)
        return out

    def replace_followup_strategies(
        self,
        site: str,
        source_strategies: list[dict[str, Any]],
        replay_strategies: list[dict[str, Any]],
    ) -> int:
        """한 후보가 만든 후속 행동 전략을 승인된 집합으로 교체한다."""

        candidate_keys = sorted(
            {
                self._followup_key_for_strategy(strategy)
                for strategy in [
                    *(source_strategies or []),
                    *(replay_strategies or []),
                ]
                if (
                    isinstance(strategy, dict)
                    and str(strategy.get("action") or "")
                    in _FOLLOWUP_ACTIONS
                    and isinstance(strategy.get("trigger"), dict)
                    and str(strategy["trigger"].get("action") or "")
                )
            }
        )
        if candidate_keys:
            placeholders = ",".join("?" for _ in candidate_keys)
            with self._conn() as conn:
                conn.execute(
                    "DELETE FROM followup_action_strategies "
                    f"WHERE site=? AND strategy_key IN ({placeholders})",
                    (site, *candidate_keys),
                )

        saved = 0
        now = datetime.now().isoformat(timespec="seconds")
        for raw_strategy in replay_strategies or []:
            strategy = self._validated_followup_strategy(raw_strategy)
            if strategy is None:
                continue
            strategy_key = self._followup_key_for_strategy(strategy)
            task_category = normalize_task_category(
                strategy.get("task_category")
            )
            payload = self._dump_json(strategy)
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT 1 FROM followup_action_strategies "
                    "WHERE strategy_key=?",
                    (strategy_key,),
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE followup_action_strategies "
                        "SET site=?, task_category=?, strategy_json=?, "
                        "success_count=success_count+1, updated_at=? "
                        "WHERE strategy_key=?",
                        (
                            site,
                            task_category,
                            payload,
                            now,
                            strategy_key,
                        ),
                    )
                else:
                    conn.execute(
                        "INSERT INTO followup_action_strategies "
                        "(strategy_key, site, task_category, strategy_json, "
                        "success_count, created_at, updated_at) "
                        "VALUES (?,?,?,?,1,?,?)",
                        (
                            strategy_key,
                            site,
                            task_category,
                            payload,
                            now,
                            now,
                        ),
                    )
            saved += 1
        return saved

    def get_followup_strategy(
        self,
        site: str,
        *,
        task_category: str | None,
        trigger_action: str,
        trigger_component: str = "",
        trigger_page_role: str = "",
        page_role: str = "",
        current_url_template: str = "",
    ) -> tuple[str, FollowupActionStrategy] | None:
        """현재 직전 행동 문맥과 정확히 맞는 후속 행동 전략 하나를 반환한다."""

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT strategy_key, strategy_json, success_count "
                "FROM followup_action_strategies "
                "WHERE site=? ORDER BY success_count DESC, updated_at DESC",
                (site,),
            ).fetchall()

        requested_task = normalize_task_category(task_category)
        requested_component = str(trigger_component or "").strip()
        requested_trigger_role = normalize_page_role(trigger_page_role)
        requested_page_role = normalize_page_role(page_role)
        for row in rows:
            try:
                strategy = FollowupActionStrategy(
                    **json.loads(row["strategy_json"] or "{}")
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not task_category_matches(
                requested_task,
                strategy.task_category,
            ):
                continue
            if strategy.trigger.action != trigger_action:
                continue
            if (
                requested_component
                and strategy.trigger.component
                and strategy.trigger.component != requested_component
            ):
                continue
            if (
                requested_trigger_role
                and strategy.trigger.page_role
                and normalize_page_role(strategy.trigger.page_role)
                != requested_trigger_role
            ):
                continue
            if (
                requested_page_role
                and strategy.page_role
                and normalize_page_role(strategy.page_role)
                != requested_page_role
            ):
                continue
            if (
                current_url_template
                and strategy.url_template
                and strategy.url_template != current_url_template
            ):
                continue
            strategy.success_count = int(row["success_count"] or 0)
            return str(row["strategy_key"] or ""), strategy
        return None

    def active_counts(self, site: str | None = None) -> dict[str, int]:
        """E2E 사전조건 검사용 활성 자동화 데이터 개수를 반환한다."""

        where = " WHERE site=?" if site else ""
        params = (site,) if site else ()
        with self._conn() as conn:
            recipe_count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM recipes{where}",
                    params,
                ).fetchone()[0]
            )
            followup_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM followup_action_strategies"
                    + where,
                    params,
                ).fetchone()[0]
            )
        return {
            "roi_recipes": recipe_count,
            "followup_strategies": followup_count,
            "total": recipe_count + followup_count,
        }

    def get_site_recipes(self, site: str, *, task_category: str | None = None) -> list[tuple[str, SiteRecipe]]:
        """같은 사이트의 활성 레시피를 성공 횟수 순으로 반환한다."""
        candidates: list[tuple[str, SiteRecipe]] = []
        for item in self.get_by_site(site):
            steps = list(item.get("steps") or [])
            if not steps:
                continue
            metadata = RecipeSkillMetadata(**(item.get("skill_metadata") or {}))
            if not self._metadata_matches_task_category(metadata, task_category):
                continue
            recipe = SiteRecipe(
                site=item.get("site") or site,
                goal=item.get("goal") or "",
                steps=[RecipeStep(**step) for step in steps],
                skill_metadata=metadata,
                success_count=item.get("success_count") or 0,
            )
            candidates.append((item.get("recipe_key") or "", recipe))
        return candidates
