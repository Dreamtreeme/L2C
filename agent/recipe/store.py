"""검증된 경험 경로를 SQLite에 저장하고 사이트별로 조회한다."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from agent.recipe.sqlite_store import SQLiteStore
from agent.recipe.task_category import normalize_task_category, task_category_matches
from agent.runtime.site_context import normalize_page_role
from agent.utils.text import normalize_text
from shared.schema.recipe_schema import (
    ExperiencePath,
    PhysicalAction,
    SiteExperience,
)
from shared.schema.skill_schema import RecipeSkillMetadata


_RECIPE_KEY_VERSION = 8
_RECIPE_KEY_PREFIX = "experience8#"


class StoredExperienceRecord(BaseModel):
    """SQLite 행을 애플리케이션에서 사용하는 타입으로 변환한 결과."""

    model_config = ConfigDict(extra="forbid")

    recipe_key: str
    site: str
    goal: str
    path: ExperiencePath
    skill_metadata: RecipeSkillMetadata
    support_count: int
    replay_success_count: int
    replay_failure_count: int
    last_replayed_at: str
    updated_at: str
    source_count: int

    def as_site_experience(self) -> SiteExperience:
        return SiteExperience(
            site=self.site,
            goal=self.goal,
            transitions=self.path.transitions,
            skill_metadata=self.skill_metadata,
            support_count=self.support_count,
            replay_success_count=self.replay_success_count,
            replay_failure_count=self.replay_failure_count,
            updated_at=self.updated_at,
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "recipe_key": self.recipe_key,
            "site": self.site,
            "goal": self.goal,
            **self.path.model_dump(mode="json", exclude_none=True),
            "skill_metadata": self.skill_metadata.model_dump(mode="json"),
            "support_count": self.support_count,
            "replay_success_count": self.replay_success_count,
            "replay_failure_count": self.replay_failure_count,
            "last_replayed_at": self.last_replayed_at,
            "updated_at": self.updated_at,
            "source_count": self.source_count,
        }


def _persistent_path(path: ExperiencePath) -> ExperiencePath:
    """실행마다 달라지는 관찰 ID와 실행 근거를 활성 경로에서 제외한다."""

    transitions = [
        transition.model_copy(
            deep=True,
            update={
                "before": transition.before.model_copy(
                    deep=True,
                    update={"observation_id": ""},
                ),
                "after": transition.after.model_copy(
                    deep=True,
                    update={"observation_id": ""},
                ),
                "evidence": None,
            },
        )
        for transition in path.transitions
    ]
    return ExperiencePath(transitions=transitions)


def _action_path_identity(action: PhysicalAction) -> dict[str, object]:
    """경로 분기를 구분하는 행동의 안정적인 의미 정보를 만든다."""

    target = action.target
    fixed_param: dict[str, str] = {}
    if action.replay_mode == "fixed":
        if action.action == "type_in_marker":
            fixed_param["text"] = normalize_text(action.param.text)
        elif action.action == "press_key":
            fixed_param["key"] = action.param.key
    return {
        "action": action.action,
        "component": action.component,
        "target_role": action.target_role,
        "slot_refs": (
            sorted(action.slot_refs)
            if action.replay_mode == "parameterized"
            else []
        ),
        "target_region": target.region if target and target.region else "",
        "target_label": normalize_text(
            (target.semantic_label or target.text) if target else ""
        ),
        "fixed_param": fixed_param,
    }


def _recipe_key_for_path(
    site: str,
    path: ExperiencePath,
    metadata: RecipeSkillMetadata | None = None,
) -> str:
    """전체 상태 전이 순서와 의미를 포함한 안정 경로 키를 만든다."""

    payload = {
        "key_version": _RECIPE_KEY_VERSION,
        "site": site,
        "task_category": normalize_task_category(
            metadata.task_category if metadata else ""
        ),
        "path": [
            {
                "before_url": transition.before.url_template,
                "before_role": normalize_page_role(
                    transition.before.page_role
                ),
                "actions": [
                    _action_path_identity(action)
                    for action in transition.actions
                ],
                "after_url": transition.after.url_template,
            }
            for transition in path.transitions
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
                    support_count INTEGER NOT NULL DEFAULT 0,
                    replay_success_count INTEGER NOT NULL DEFAULT 0,
                    replay_failure_count INTEGER NOT NULL DEFAULT 0,
                    last_replayed_at TEXT,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recipes_site ON recipes(site)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recipe_sources (
                    recipe_key TEXT NOT NULL,
                    run_id     TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (recipe_key, run_id),
                    FOREIGN KEY (recipe_key)
                        REFERENCES recipes(recipe_key) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recipe_sources_run "
                "ON recipe_sources(run_id)"
            )

    def _upsert_recipe_path(
        self,
        site: str,
        goal: str,
        path: ExperiencePath,
        metadata: RecipeSkillMetadata | None = None,
        source_run_id: str = "",
    ) -> bool:
        """같은 의미의 전체 안정 경로를 저장하거나 갱신한다."""

        replay_path = _persistent_path(path)
        recipe_key = _recipe_key_for_path(site, replay_path, metadata)

        now = datetime.now().isoformat(timespec="seconds")
        path_payload = replay_path.model_dump_json(exclude_none=True)
        metadata_payload = self.dump_json(
            metadata.model_dump(mode="json") if metadata else {}
        )
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM recipes WHERE recipe_key=?", (recipe_key,)
            ).fetchone()
            source_exists = bool(
                source_run_id
                and conn.execute(
                    "SELECT 1 FROM recipe_sources "
                    "WHERE recipe_key=? AND run_id=?",
                    (recipe_key, source_run_id),
                ).fetchone()
            )
            if row:
                conn.execute(
                    "UPDATE recipes "
                    "SET site=?, goal=?, path_json=?, metadata_json=?, "
                    "support_count=support_count+?, updated_at=? "
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
                    "support_count, created_at, updated_at) "
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
            if source_run_id and not source_exists:
                conn.execute(
                    "INSERT INTO recipe_sources "
                    "(recipe_key, run_id, created_at) VALUES (?,?,?)",
                    (recipe_key, source_run_id, now),
                )
        return True

    def commit_recipe_path(
        self,
        site: str,
        goal: str,
        path: ExperiencePath,
        metadata: RecipeSkillMetadata | None = None,
        source_run_id: str = "",
    ) -> int:
        """성공 후보의 상태 전이 경로 하나를 저장한다."""

        return int(
            self._upsert_recipe_path(
                site,
                goal,
                path,
                metadata=metadata,
                source_run_id=source_run_id,
            )
        )

    def _detach_run_paths(self, run_id: str) -> None:
        """후보 근거를 떼고 다른 근거가 없는 경로만 제거한다."""

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT recipe_key FROM recipe_sources WHERE run_id=?",
                (run_id,),
            ).fetchall()
            previous_keys = [
                str(row["recipe_key"]) for row in rows if str(row["recipe_key"])
            ]
            conn.execute("DELETE FROM recipe_sources WHERE run_id=?", (run_id,))
            if not previous_keys:
                return
            placeholders = ",".join("?" for _ in previous_keys)
            conn.execute(
                "UPDATE recipes "
                "SET support_count=MAX(0, support_count-1) "
                f"WHERE recipe_key IN ({placeholders})",
                previous_keys,
            )
            conn.execute(
                "DELETE FROM recipes "
                f"WHERE recipe_key IN ({placeholders}) "
                "AND support_count<=0 "
                "AND NOT EXISTS ("
                "SELECT 1 FROM recipe_sources "
                "WHERE recipe_sources.recipe_key=recipes.recipe_key"
                ")",
                previous_keys,
            )

    def clear_recipes(self, site: str | None = None) -> int:
        """활성 경험 경로만 비우고 후보 증거는 보존한다."""

        with self._conn() as conn:
            if site:
                result = conn.execute("DELETE FROM recipes WHERE site=?", (site,))
            else:
                result = conn.execute("DELETE FROM recipes")
            return int(result.rowcount or 0)

    def replace_recipe_paths(
        self,
        site: str,
        goal: str,
        recipe_paths: list[ExperiencePath],
        metadata: RecipeSkillMetadata | None = None,
        source_run_id: str = "",
    ) -> int:
        """한 후보가 소유한 기존 경로만 지우고 새 안정 경로로 교체한다."""

        if source_run_id:
            self._detach_run_paths(source_run_id)
        return sum(
            self.commit_recipe_path(
                site,
                goal,
                path,
                metadata=metadata,
                source_run_id=source_run_id,
            )
            for path in recipe_paths
        )

    def record_replay_result(self, recipe_key: str, succeeded: bool) -> bool:
        """경험 기반 경로 전체의 실제 재생 결과를 누적한다."""

        column = "replay_success_count" if succeeded else "replay_failure_count"
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            result = conn.execute(
                f"UPDATE recipes SET {column}={column}+1, "
                "last_replayed_at=?, updated_at=? WHERE recipe_key=?",
                (now, now, recipe_key),
            )
        return bool(result.rowcount)

    def _site_records(self, site: str) -> list[StoredExperienceRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT recipe_key, site, goal, path_json, metadata_json, "
                "support_count, replay_success_count, replay_failure_count, "
                "last_replayed_at, updated_at, "
                "(SELECT COUNT(*) FROM recipe_sources "
                "WHERE recipe_sources.recipe_key=recipes.recipe_key) "
                "AS source_count FROM recipes "
                "WHERE site=? AND recipe_key LIKE ? "
                "ORDER BY replay_success_count DESC, "
                "replay_failure_count ASC, support_count DESC, "
                "updated_at DESC, recipe_key ASC",
                (site, f"{_RECIPE_KEY_PREFIX}%"),
            ).fetchall()
        return [
            StoredExperienceRecord(
                recipe_key=str(row["recipe_key"]),
                site=str(row["site"]),
                goal=str(row["goal"] or ""),
                path=ExperiencePath.model_validate_json(row["path_json"]),
                skill_metadata=RecipeSkillMetadata.model_validate_json(
                    row["metadata_json"] or "{}"
                ),
                support_count=int(row["support_count"] or 0),
                replay_success_count=int(row["replay_success_count"] or 0),
                replay_failure_count=int(row["replay_failure_count"] or 0),
                last_replayed_at=str(row["last_replayed_at"] or ""),
                updated_at=str(row["updated_at"] or ""),
                source_count=int(row["source_count"] or 0),
            )
            for row in rows
        ]

    def get_by_site(self, site: str) -> list[dict[str, object]]:
        return [record.as_payload() for record in self._site_records(site)]

    def active_counts(self, site: str | None = None) -> dict[str, int]:
        """E2E 사전조건 검사용 활성 자동화 데이터 개수를 반환한다."""

        where = " WHERE recipe_key LIKE ?"
        params: tuple[str, ...] = (f"{_RECIPE_KEY_PREFIX}%",)
        if site:
            where += " AND site=?"
            params += (site,)
        with self._conn() as conn:
            recipe_count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM recipes{where}",
                    params,
                ).fetchone()[0]
            )
        return {"roi_recipes": recipe_count, "total": recipe_count}

    def get_site_recipes(
        self,
        site: str,
        *,
        task_category: str | None = None,
    ) -> list[tuple[str, SiteExperience]]:
        """같은 사이트의 활성 경험을 실제 재생 결과 순으로 반환한다."""

        return [
            (record.recipe_key, record.as_site_experience())
            for record in self._site_records(site)
            if task_category_matches(
                task_category,
                record.skill_metadata.task_category,
            )
        ]


__all__ = ["RecipeStore", "StoredExperienceRecord"]
