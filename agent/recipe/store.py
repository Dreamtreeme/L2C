"""반사 레시피 저장소(RecipeStore).

채용공고 DB(jobs DB)와 같은 SQLite 파일에 ROI 검증용 레시피를 저장한다.
활성 레시피 조회는 화면 상태 해시가 아니라 사이트/작업분류 후보 집합과 ROI 검증으로 한다.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from agent.recipe.page_context import normalize_page_role
from agent.recipe.payload_sanitizer import strip_replay_runtime_fields
from agent.recipe.replay_actions import (
    CONTEXTUAL_REPLAY_ACTIONS,
    TARGET_REPLAY_ACTIONS,
    is_supported_recipe_action_group,
)
from agent.recipe.sqlite_store import SQLiteStore
from agent.recipe.task_category import normalize_task_category, task_category_matches
from agent.recipe.text_utils import normalize_text
from agent.utils.model_dump import dump_model
from shared.schema.recipe_schema import RecipePath, SiteRecipe
from shared.schema.skill_schema import RecipeSkillMetadata


_RECIPE_KEY_VERSION = 6
_RECIPE_KEY_PREFIX = "path6#"


class RecipeStore(SQLiteStore):
    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recipes (
                    recipe_key    TEXT PRIMARY KEY,
                    site          TEXT NOT NULL,
                    goal          TEXT,
                    path_json     TEXT NOT NULL,
                    metadata_json TEXT,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_recipes_site ON recipes(site)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recipe_sources (
                    recipe_key   TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    PRIMARY KEY (recipe_key, candidate_id),
                    FOREIGN KEY (recipe_key)
                        REFERENCES recipes(recipe_key) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recipe_sources_candidate "
                "ON recipe_sources(candidate_id)"
            )

    @staticmethod
    def _action_has_required_replay_fields(
        action_item: dict[str, Any],
    ) -> bool:
        action = str(action_item.get("action") or "")
        if action in TARGET_REPLAY_ACTIONS:
            return bool(
                action_item.get("target")
                and action_item.get("roi_signature")
            )
        if action in CONTEXTUAL_REPLAY_ACTIONS:
            param = (
                action_item.get("param")
                if isinstance(action_item.get("param"), dict)
                else {}
            )
            if str(action_item.get("replay_mode") or "") != "fixed":
                return False
            if action == "press_key":
                return bool(param.get("key"))
            if action == "switch_tab":
                return bool(param.get("direction"))
            return True
        return False

    @classmethod
    def _transition_has_required_replay_fields(
        cls,
        transition: dict[str, Any],
    ) -> bool:
        actions = [
            item
            for item in transition.get("actions", []) or []
            if isinstance(item, dict)
        ]
        if not actions or not is_supported_recipe_action_group(actions):
            return False
        if not all(
            cls._action_has_required_replay_fields(item)
            for item in actions
        ):
            return False
        before = (
            transition.get("before")
            if isinstance(transition.get("before"), dict)
            else {}
        )
        after = (
            transition.get("after")
            if isinstance(transition.get("after"), dict)
            else {}
        )
        if not before or not after:
            return False
        if after.get("anchor_target") and after.get(
            "anchor_roi_signature"
        ):
            return True
        if dict(after.get("screen_context_signature") or {}).get("phash"):
            return True
        return bool(
            after.get("url_template")
            and after.get("url_template") != before.get("url_template")
        )

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
    def _action_path_identity(
        action_item: dict[str, Any],
    ) -> dict[str, Any]:
        """경로 분기를 구분하는 행동의 안정적인 의미 정보를 만든다."""

        target = (
            action_item.get("target")
            if isinstance(action_item.get("target"), dict)
            else {}
        )
        replay_mode = str(action_item.get("replay_mode") or "")
        param = (
            action_item.get("param")
            if isinstance(action_item.get("param"), dict)
            else {}
        )
        action = str(action_item.get("action") or "")
        fixed_param: dict[str, Any] = {}
        if replay_mode == "fixed":
            if action == "type_in_marker":
                fixed_param["text"] = normalize_text(
                    param.get("text") or action_item.get("value")
                )
            elif action == "press_key":
                fixed_param["key"] = str(param.get("key") or "")
            elif action == "switch_tab":
                fixed_param["direction"] = str(
                    param.get("direction") or ""
                )
        return {
            "action": action,
            "component": action_item.get("component") or "",
            "target_role": action_item.get("target_role") or "",
            "slot_refs": (
                sorted(
                    str(item)
                    for item in action_item.get("slot_refs") or []
                )
                if replay_mode == "parameterized"
                else []
            ),
            "target_region": target.get("region") or "",
            "target_label": normalize_text(
                target.get("semantic_label") or target.get("text")
            ),
            "fixed_param": fixed_param,
        }

    @staticmethod
    def _recipe_key_for_path(
        site: str,
        path: dict[str, Any],
        metadata: dict[str, Any] | RecipeSkillMetadata | None = None,
    ) -> str:
        """전체 상태 전이 순서와 의미를 포함한 안정 경로 키를 만든다."""

        meta = RecipeStore._metadata_dict(metadata)
        transitions = [
            item
            for item in path.get("transitions", []) or []
            if isinstance(item, dict)
        ]
        payload = {
            "key_version": _RECIPE_KEY_VERSION,
            "site": site or "",
            "task_category": normalize_task_category(
                meta.get("task_category")
            ),
            "path": [
                {
                    "before_url": str(
                        (transition.get("before") or {}).get(
                            "url_template"
                        )
                        or ""
                    ),
                    "before_role": normalize_page_role(
                        (transition.get("before") or {}).get("page_role")
                    ),
                    "actions": [
                        RecipeStore._action_path_identity(action_item)
                        for action_item in transition.get("actions", []) or []
                        if isinstance(action_item, dict)
                    ],
                    "after_url": str(
                        (transition.get("after") or {}).get(
                            "url_template"
                        )
                        or ""
                    ),
                }
                for transition in transitions
            ],
        }
        digest = hashlib.sha1(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        return f"{_RECIPE_KEY_PREFIX}{digest}"

    def _upsert_recipe_path(
        self,
        site: str,
        goal: str,
        path: dict[str, Any],
        metadata: dict[str, Any] | RecipeSkillMetadata | None = None,
        candidate_id: str = "",
    ) -> bool:
        """같은 의미의 전체 안정 경로를 저장하거나 갱신한다."""

        if not isinstance(path, dict):
            return False
        try:
            replay_path = dump_model(
                RecipePath.model_validate(
                    strip_replay_runtime_fields(path)
                )
            )
        except (TypeError, ValueError):
            return False
        transitions = [
            item
            for item in replay_path.get("transitions", []) or []
            if isinstance(item, dict)
        ]
        if (
            not replay_path.get("start_state")
            or not replay_path.get("completion_state")
            or len(transitions)
            != len(replay_path.get("transitions", []) or [])
            or not all(
                self._transition_has_required_replay_fields(transition)
                for transition in transitions
            )
        ):
            return False
        first_actions = transitions[0].get("actions", []) or []
        if (
            not first_actions
            or str(first_actions[0].get("action") or "")
            not in TARGET_REPLAY_ACTIONS
        ):
            return False
        recipe_key = self._recipe_key_for_path(
            site,
            replay_path,
            metadata=metadata,
        )

        now = datetime.now().isoformat(timespec="seconds")
        path_payload = json.dumps(replay_path, ensure_ascii=False)
        metadata_payload = self.dump_json(self._metadata_dict(metadata))
        with self._conn() as conn:
            row = conn.execute("SELECT 1 FROM recipes WHERE recipe_key=?", (recipe_key,)).fetchone()
            source_exists = bool(
                candidate_id
                and conn.execute(
                    "SELECT 1 FROM recipe_sources "
                    "WHERE recipe_key=? AND candidate_id=?",
                    (recipe_key, candidate_id),
                ).fetchone()
            )
            if row:
                conn.execute(
                    "UPDATE recipes "
                    "SET site=?, goal=?, path_json=?, metadata_json=?, "
                    "success_count=success_count+?, updated_at=? "
                    "WHERE recipe_key=?",
                    (
                        site,
                        goal,
                        path_payload,
                        metadata_payload,
                        0 if source_exists else 1,
                        now,
                        recipe_key,
                    ),
                )
            else:
                conn.execute(
                    "INSERT INTO recipes "
                    "(recipe_key, site, goal, path_json, metadata_json, "
                    "success_count, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,1,?,?)",
                    (
                        recipe_key,
                        site,
                        goal,
                        path_payload,
                        metadata_payload,
                        now,
                        now,
                    ),
                )
            if candidate_id and not source_exists:
                conn.execute(
                    "INSERT INTO recipe_sources "
                    "(recipe_key, candidate_id, created_at) VALUES (?,?,?)",
                    (recipe_key, candidate_id, now),
                )
        return True

    def commit_recipe_path(
        self,
        site: str,
        goal: str,
        path: dict[str, Any],
        metadata: dict[str, Any] | RecipeSkillMetadata | None = None,
        candidate_id: str = "",
    ) -> int:
        """성공 후보의 상태 전이 경로 하나를 저장한다."""

        return int(
            self._upsert_recipe_path(
                site,
                goal,
                path,
                metadata=metadata,
                candidate_id=candidate_id,
            )
        )

    def _detach_candidate_paths(self, candidate_id: str) -> None:
        """후보 근거를 떼고 다른 근거가 없는 경로만 제거한다."""

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT recipe_key FROM recipe_sources "
                "WHERE candidate_id=?",
                (candidate_id,),
            ).fetchall()
            previous_keys = [
                str(row["recipe_key"])
                for row in rows
                if str(row["recipe_key"])
            ]
            conn.execute(
                "DELETE FROM recipe_sources WHERE candidate_id=?",
                (candidate_id,),
            )
            if not previous_keys:
                return
            placeholders = ",".join("?" for _ in previous_keys)
            conn.execute(
                "UPDATE recipes "
                "SET success_count=MAX(0, success_count-1) "
                f"WHERE recipe_key IN ({placeholders})",
                previous_keys,
            )
            conn.execute(
                "DELETE FROM recipes "
                f"WHERE recipe_key IN ({placeholders}) "
                "AND success_count<=0 "
                "AND NOT EXISTS ("
                "SELECT 1 FROM recipe_sources "
                "WHERE recipe_sources.recipe_key=recipes.recipe_key"
                ")",
                previous_keys,
            )

    def clear_recipes(self, site: str | None = None) -> int:
        """활성 ROI 레시피만 비우고 후보 증거는 보존한다."""

        with self._conn() as conn:
            if site:
                recipe_result = conn.execute(
                    "DELETE FROM recipes WHERE site=?",
                    (site,),
                )
            else:
                recipe_result = conn.execute("DELETE FROM recipes")
            return int(recipe_result.rowcount or 0)

    def replace_recipe_paths(
        self,
        site: str,
        goal: str,
        recipe_paths: list[dict[str, Any]],
        metadata: dict[str, Any] | RecipeSkillMetadata | None = None,
        candidate_id: str = "",
    ) -> int:
        """한 후보가 소유한 기존 경로만 지우고 새 안정 경로로 교체한다."""

        if candidate_id:
            self._detach_candidate_paths(candidate_id)

        saved = 0
        for path in recipe_paths or []:
            saved += self.commit_recipe_path(
                site,
                goal,
                path,
                metadata=metadata,
                candidate_id=candidate_id,
            )
        return saved

    def get_by_site(self, site: str):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT recipe_key, site, goal, path_json, metadata_json, "
                "success_count, "
                "(SELECT COUNT(*) FROM recipe_sources "
                "WHERE recipe_sources.recipe_key=recipes.recipe_key) "
                "AS source_count FROM recipes "
                "WHERE site=? "
                "ORDER BY success_count DESC, updated_at DESC, recipe_key ASC",
                (site,),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            path = json.loads(item.pop("path_json") or "{}")
            item.update(path)
            item["skill_metadata"] = json.loads(item.pop("metadata_json") or "{}")
            out.append(item)
        return out

    def active_counts(self, site: str | None = None) -> dict[str, int]:
        """E2E 사전조건 검사용 활성 자동화 데이터 개수를 반환한다."""

        where = ""
        params: tuple[Any, ...] = ()
        if site:
            where = " WHERE site=?"
            params += (site,)
        with self._conn() as conn:
            recipe_count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM recipes{where}",
                    params,
                ).fetchone()[0]
            )
        return {
            "roi_recipes": recipe_count,
            "total": recipe_count,
        }

    def get_site_recipes(self, site: str, *, task_category: str | None = None) -> list[tuple[str, SiteRecipe]]:
        """같은 사이트의 활성 레시피를 성공 횟수 순으로 반환한다."""
        candidates: list[tuple[str, SiteRecipe]] = []
        for item in self.get_by_site(site):
            transitions = list(item.get("transitions") or [])
            if not transitions:
                continue
            metadata = RecipeSkillMetadata(**(item.get("skill_metadata") or {}))
            if not self._metadata_matches_task_category(metadata, task_category):
                continue
            recipe = SiteRecipe(
                site=item.get("site") or site,
                goal=item.get("goal") or "",
                start_state=item.get("start_state") or {},
                transitions=transitions,
                completion_state=item.get("completion_state") or {},
                skill_metadata=metadata,
                success_count=item.get("success_count") or 0,
            )
            candidates.append((item.get("recipe_key") or "", recipe))
        return candidates


__all__ = ["RecipeStore"]
