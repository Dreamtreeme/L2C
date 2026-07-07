"""반사 레시피 저장소(RecipeStore).

채용공고 DB(jobs DB)와 같은 SQLite 파일에 화면 상태 키(state_key) 기준으로
재생할 행동 단계와 스킬형 메타데이터(skill_metadata)를 저장한다.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.utils.model_dump import dump_model
from shared.schema.recipe_schema import RecipeStep, SiteRecipe
from shared.schema.skill_schema import RecipeSkillMetadata


class RecipeStore:
    def __init__(self, db_path=None):
        if db_path is None:
            # 설정 지연 로드(lazy import)로 모듈 import 시점의 부작용을 줄인다.
            from shared.config import DB_PATH

            db_path = DB_PATH
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

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
                columns = [
                    item["name"]
                    for item in conn.execute("PRAGMA table_info(recipes)").fetchall()
                ]
                if "state_key" not in columns:
                    legacy_name = f"recipes_legacy_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    conn.execute(f"ALTER TABLE recipes RENAME TO {legacy_name}")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recipes (
                    state_key     TEXT PRIMARY KEY,
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
            columns = {
                item["name"]
                for item in conn.execute("PRAGMA table_info(recipes)").fetchall()
            }
            if "metadata_json" not in columns:
                conn.execute("ALTER TABLE recipes ADD COLUMN metadata_json TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_recipes_site ON recipes(site)")

    @staticmethod
    def _dump_json(payload: Any) -> str:
        return json.dumps(payload or {}, ensure_ascii=False)

    @staticmethod
    def _metadata_dict(metadata: dict[str, Any] | RecipeSkillMetadata | None) -> dict[str, Any]:
        return dump_model(metadata)

    def record_step(
        self,
        site: str,
        goal: str,
        step_or_steps,
        metadata: dict[str, Any] | RecipeSkillMetadata | None = None,
    ) -> bool:
        """같은 화면 상태(state_key)의 행동 묶음을 저장하거나 갱신한다."""

        steps = list(step_or_steps if isinstance(step_or_steps, list) else [step_or_steps])
        steps = [step for step in steps if isinstance(step, dict)]
        if not steps:
            return False
        state_key = steps[0].get("state_key")
        if not state_key:
            return False

        now = datetime.now().isoformat(timespec="seconds")
        steps_payload = json.dumps(steps, ensure_ascii=False)
        metadata_payload = self._dump_json(self._metadata_dict(metadata))
        with self._conn() as conn:
            row = conn.execute("SELECT 1 FROM recipes WHERE state_key=?", (state_key,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE recipes "
                    "SET site=?, goal=?, steps_json=?, metadata_json=?, success_count=success_count+1, updated_at=? "
                    "WHERE state_key=?",
                    (site, goal, steps_payload, metadata_payload, now, state_key),
                )
            else:
                conn.execute(
                    "INSERT INTO recipes "
                    "(state_key, site, goal, steps_json, metadata_json, success_count, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,1,?,?)",
                    (state_key, site, goal, steps_payload, metadata_payload, now, now),
                )
        return True

    def commit_recipe(
        self,
        site: str,
        goal: str,
        steps: list[dict],
        metadata: dict[str, Any] | RecipeSkillMetadata | None = None,
    ) -> int:
        """성공 후보의 행동을 상태 키(state_key)별 실행 묶음으로 저장한다."""

        saved = 0
        groups: list[list[dict]] = []
        for step in steps or []:
            if not isinstance(step, dict) or not step.get("state_key"):
                continue
            if not groups or groups[-1][0].get("state_key") != step.get("state_key"):
                groups.append([])
            groups[-1].append(dict(step))

        for group in groups:
            if self.record_step(site, goal, group, metadata=metadata):
                saved += 1
        return saved

    def replace_recipe_steps(
        self,
        site: str,
        goal: str,
        source_steps: list[dict],
        replay_steps: list[dict],
        metadata: dict[str, Any] | RecipeSkillMetadata | None = None,
    ) -> int:
        """한 후보가 소유한 기존 상태 행을 지우고 Critic이 승인한 재생 단계만 교체 저장한다."""
        state_keys = sorted(
            {
                str(step.get("state_key") or "")
                for step in source_steps or []
                if isinstance(step, dict) and step.get("state_key")
            }
        )
        if state_keys:
            placeholders = ",".join("?" for _ in state_keys)
            with self._conn() as conn:
                conn.execute(
                    f"DELETE FROM recipes WHERE site=? AND state_key IN ({placeholders})",
                    (site, *state_keys),
                )
        return self.commit_recipe(site, goal, replay_steps, metadata=metadata)

    def get_recipe(self, state_key: str) -> SiteRecipe | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT state_key, site, goal, steps_json, metadata_json, success_count, updated_at "
                "FROM recipes WHERE state_key=?",
                (state_key,),
            ).fetchone()
        if not row:
            return None
        steps = [RecipeStep(**step) for step in json.loads(row["steps_json"] or "[]")]
        metadata = RecipeSkillMetadata(**json.loads(row["metadata_json"] or "{}"))
        return SiteRecipe(
            site=row["site"],
            goal=row["goal"] or "",
            steps=steps,
            skill_metadata=metadata,
            success_count=row["success_count"] or 0,
            updated_at=row["updated_at"] or "",
        )

    def get_by_site(self, site: str):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT state_key, site, goal, steps_json, metadata_json, success_count FROM recipes "
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

    def get_site_recipes(self, site: str) -> list[tuple[str, SiteRecipe]]:
        """같은 사이트의 활성 레시피를 성공 횟수 순으로 반환한다."""
        candidates: list[tuple[str, SiteRecipe]] = []
        for item in self.get_by_site(site):
            steps = list(item.get("steps") or [])
            if not steps:
                continue
            recipe = SiteRecipe(
                site=item.get("site") or site,
                goal=item.get("goal") or "",
                steps=[RecipeStep(**step) for step in steps],
                skill_metadata=RecipeSkillMetadata(**(item.get("skill_metadata") or {})),
                success_count=item.get("success_count") or 0,
            )
            candidates.append((item.get("state_key") or "", recipe))
        return candidates
