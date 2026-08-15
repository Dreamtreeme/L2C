"""검증된 경험 규칙을 SQLite에 저장하고 사이트별로 조회한다."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from agent.recipe.sqlite_store import SQLiteStore
from agent.recipe.task_category import normalize_task_category, task_category_matches
from agent.utils.text import normalize_text
from shared.schema.experience_rule_schema import ExperienceRule
from shared.schema.skill_schema import RecipeSkillMetadata


_RULE_KEY_VERSION = 10
_RULE_KEY_PREFIX = "experience-rule10#"


class StoredExperienceRule(BaseModel):
    """SQLite 행과 실제 재생 통계를 결합한 경험 규칙."""

    model_config = ConfigDict(extra="forbid")

    rule_key: str
    rule: ExperienceRule
    support_count: int
    replay_success_count: int
    replay_failure_count: int
    last_replayed_at: str
    updated_at: str
    source_count: int

    def active_rule(self) -> ExperienceRule:
        return self.rule.model_copy(
            deep=True,
            update={
                "support_count": self.support_count,
                "replay_success_count": self.replay_success_count,
                "replay_failure_count": self.replay_failure_count,
                "updated_at": self.updated_at,
            },
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "rule_key": self.rule_key,
            **self.active_rule().model_dump(mode="json", exclude_none=True),
            "last_replayed_at": self.last_replayed_at,
            "source_count": self.source_count,
        }


def _persistent_rule(rule: ExperienceRule) -> ExperienceRule:
    return rule.model_copy(
        deep=True,
        update={
            "support_count": 1,
            "replay_success_count": 0,
            "replay_failure_count": 0,
            "updated_at": "",
        },
    )


def _rule_key_for_purpose(site: str, metadata: RecipeSkillMetadata) -> str:
    normalized_site = normalize_text(site).casefold()
    task_category = normalize_task_category(metadata.task_category)
    if not normalized_site or not task_category:
        raise ValueError("경험 규칙 저장에는 site와 task_category가 필요합니다.")
    payload = {
        "key_version": _RULE_KEY_VERSION,
        "site": normalized_site,
        "task_category": task_category,
    }
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{_RULE_KEY_PREFIX}{digest}"


class ExperienceRuleStore(SQLiteStore):
    """활성 경험 규칙의 단일 저장소."""

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experience_rules (
                    rule_key      TEXT PRIMARY KEY,
                    site          TEXT NOT NULL,
                    goal          TEXT,
                    rule_json     TEXT NOT NULL,
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
                "CREATE INDEX IF NOT EXISTS idx_experience_rules_site "
                "ON experience_rules(site)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experience_rule_sources (
                    rule_key  TEXT NOT NULL,
                    run_id    TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (rule_key, run_id),
                    FOREIGN KEY (rule_key)
                        REFERENCES experience_rules(rule_key) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experience_rule_sources_run "
                "ON experience_rule_sources(run_id)"
            )
            conn.execute("DROP TABLE IF EXISTS recipe_sources")
            conn.execute("DROP TABLE IF EXISTS recipes")

    def save_rule(
        self,
        rule: ExperienceRule,
        *,
        source_run_id: str = "",
    ) -> int:
        """같은 사이트·작업 목적의 규칙을 최신 검증 결과로 저장한다."""

        persistent = _persistent_rule(rule)
        rule_key = _rule_key_for_purpose(rule.site, rule.skill_metadata)
        payload = persistent.model_dump_json(exclude_none=True)
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            row = conn.execute(
                "SELECT rule_json FROM experience_rules WHERE rule_key=?",
                (rule_key,),
            ).fetchone()
            source_exists = bool(
                source_run_id
                and conn.execute(
                    "SELECT 1 FROM experience_rule_sources "
                    "WHERE rule_key=? AND run_id=?",
                    (rule_key, source_run_id),
                ).fetchone()
            )
            if row and str(row["rule_json"]) == payload:
                conn.execute(
                    "UPDATE experience_rules SET goal=?, "
                    "support_count=support_count+?, updated_at=? WHERE rule_key=?",
                    (rule.goal, 0 if source_exists else 1, now, rule_key),
                )
            elif row:
                conn.execute(
                    "DELETE FROM experience_rule_sources WHERE rule_key=?",
                    (rule_key,),
                )
                conn.execute(
                    "UPDATE experience_rules SET site=?, goal=?, rule_json=?, "
                    "support_count=1, replay_success_count=0, "
                    "replay_failure_count=0, last_replayed_at=NULL, updated_at=? "
                    "WHERE rule_key=?",
                    (rule.site, rule.goal, payload, now, rule_key),
                )
                source_exists = False
            else:
                conn.execute(
                    "INSERT INTO experience_rules "
                    "(rule_key, site, goal, rule_json, support_count, created_at, "
                    "updated_at) VALUES (?,?,?,?,1,?,?)",
                    (rule_key, rule.site, rule.goal, payload, now, now),
                )
            if source_run_id and not source_exists:
                conn.execute(
                    "INSERT INTO experience_rule_sources "
                    "(rule_key, run_id, created_at) VALUES (?,?,?)",
                    (rule_key, source_run_id, now),
                )
        return 1

    def clear_rules(self, site: str | None = None) -> int:
        """활성 경험 규칙만 비우고 자율탐색 후보는 보존한다."""

        with self._conn() as conn:
            result = (
                conn.execute("DELETE FROM experience_rules WHERE site=?", (site,))
                if site
                else conn.execute("DELETE FROM experience_rules")
            )
            return int(result.rowcount or 0)

    def record_replay_result(self, rule_key: str, succeeded: bool) -> bool:
        column = "replay_success_count" if succeeded else "replay_failure_count"
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            result = conn.execute(
                f"UPDATE experience_rules SET {column}={column}+1, "
                "last_replayed_at=?, updated_at=? WHERE rule_key=?",
                (now, now, rule_key),
            )
        return bool(result.rowcount)

    def _site_records(self, site: str) -> list[StoredExperienceRule]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT rule_key, rule_json, support_count, "
                "replay_success_count, replay_failure_count, last_replayed_at, "
                "updated_at, (SELECT COUNT(*) FROM experience_rule_sources "
                "WHERE experience_rule_sources.rule_key=experience_rules.rule_key) "
                "AS source_count FROM experience_rules WHERE site=? "
                "ORDER BY replay_success_count DESC, replay_failure_count ASC, "
                "support_count DESC, updated_at DESC, rule_key ASC",
                (site,),
            ).fetchall()
        return [
            StoredExperienceRule(
                rule_key=str(row["rule_key"]),
                rule=ExperienceRule.model_validate_json(row["rule_json"]),
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
        where = " WHERE site=?" if site else ""
        params = (site,) if site else ()
        with self._conn() as conn:
            count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM experience_rules{where}",
                    params,
                ).fetchone()[0]
            )
        return {"experience_rules": count, "total": count}

    def get_site_rules(
        self,
        site: str,
        *,
        task_category: str | None = None,
    ) -> list[tuple[str, ExperienceRule]]:
        """같은 사이트의 활성 규칙을 실제 재생 결과 순으로 반환한다."""

        return [
            (record.rule_key, record.active_rule())
            for record in self._site_records(site)
            if task_category_matches(
                task_category,
                record.rule.skill_metadata.task_category,
            )
        ]


__all__ = ["ExperienceRuleStore", "StoredExperienceRule"]
