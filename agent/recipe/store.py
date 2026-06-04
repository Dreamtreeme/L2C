"""
레시피 저장소 (순수 sqlite3).
기존 jobs DB(DB_PATH)와 같은 파일에 state_key 기준 recipes 테이블을 둔다.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from shared.schema.recipe_schema import RecipeStep, SiteRecipe


class RecipeStore:
    def __init__(self, db_path=None):
        if db_path is None:
            from shared.config import DB_PATH  # lazy: 모듈 임포트 시점에 설정 의존 안 함
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
        with self._conn() as c:
            row = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='recipes'"
            ).fetchone()
            if row:
                columns = [
                    item["name"]
                    for item in c.execute("PRAGMA table_info(recipes)").fetchall()
                ]
                if "state_key" not in columns:
                    legacy_name = f"recipes_legacy_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    c.execute(f"ALTER TABLE recipes RENAME TO {legacy_name}")

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS recipes (
                    state_key     TEXT PRIMARY KEY,
                    site          TEXT NOT NULL,
                    goal          TEXT,
                    steps_json    TEXT NOT NULL,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                )
                """
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_recipes_site ON recipes(site)")

    def record_step(self, site: str, goal: str, step_or_steps) -> bool:
        """state_key 하나에 같은 화면에서 실행할 액션 체인을 UPSERT한다."""
        steps = list(step_or_steps if isinstance(step_or_steps, list) else [step_or_steps])
        steps = [step for step in steps if isinstance(step, dict)]
        if not steps:
            return False
        state_key = steps[0].get("state_key")
        if not state_key:
            return False
        now = datetime.now().isoformat(timespec="seconds")
        payload = json.dumps(steps, ensure_ascii=False)
        with self._conn() as c:
            row = c.execute("SELECT 1 FROM recipes WHERE state_key=?", (state_key,)).fetchone()
            if row:
                c.execute(
                    "UPDATE recipes "
                    "SET site=?, goal=?, steps_json=?, success_count=success_count+1, updated_at=? "
                    "WHERE state_key=?",
                    (site, goal, payload, now, state_key),
                )
            else:
                c.execute(
                    "INSERT INTO recipes (state_key, site, goal, steps_json, success_count, created_at, updated_at) "
                    "VALUES (?,?,?,?,1,?,?)",
                    (state_key, site, goal, payload, now, now),
                )
        return True

    def commit_recipe(self, site: str, goal: str, steps: list[dict]) -> int:
        """성공 런에서 얻은 각 state_key별 액션 체인을 저장한다."""
        saved = 0
        groups: list[list[dict]] = []
        for step in steps or []:
            if not isinstance(step, dict) or not step.get("state_key"):
                continue
            if not groups or groups[-1][0].get("state_key") != step.get("state_key"):
                groups.append([])
            groups[-1].append(dict(step))

        for idx, group in enumerate(groups):
            expected = groups[idx + 1][0].get("state_key") if idx + 1 < len(groups) else ""
            for step in group:
                if not step.get("expected_next_state"):
                    step["expected_next_state"] = expected
            if self.record_step(site, goal, group):
                saved += 1
        return saved

    def save_run(self, site: str, goal: str, steps) -> str:
        """이전 Phase0 API 호환용 래퍼."""
        self.commit_recipe(site, goal, list(steps or []))
        return (steps or [{}])[0].get("state_key", "")

    def get_recipe(self, state_key: str) -> SiteRecipe | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT state_key, site, goal, steps_json, success_count, updated_at "
                "FROM recipes WHERE state_key=?",
                (state_key,),
            ).fetchone()
        if not row:
            return None
        steps = [RecipeStep(**step) for step in json.loads(row["steps_json"] or "[]")]
        return SiteRecipe(
            site=row["site"],
            goal=row["goal"] or "",
            steps=steps,
            success_count=row["success_count"] or 0,
            updated_at=row["updated_at"] or "",
        )

    def get_by_site(self, site: str):
        with self._conn() as c:
            rows = c.execute(
                "SELECT state_key, site, goal, steps_json, success_count FROM recipes "
                "WHERE site=? ORDER BY success_count DESC",
                (site,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["steps"] = json.loads(d.pop("steps_json") or "[]")
            out.append(d)
        return out

    def get_step(self, state_key: str):
        """Phase1용: 해당 state_key에서 재생할 스텝 1개 반환(없으면 None)."""
        recipe = self.get_recipe(state_key)
        if not recipe or not recipe.steps:
            return None
        step = recipe.steps[0]
        return step.model_dump() if hasattr(step, "model_dump") else step.dict()
