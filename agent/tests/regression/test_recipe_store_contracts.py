import sqlite3

from agent.recipe.store import ExperienceRuleStore
from shared.schema.experience_rule_schema import (
    ExpectedEffect,
    ExperienceRule,
    ExperienceRuleStep,
    RuleAction,
    RuleScreen,
    RuleTarget,
)


def _rule(*, intent="검색 화면을 연다", task_category="검색"):
    return ExperienceRule(
        site="wanted",
        goal="채용공고 검색",
        skill_metadata={"task_category": task_category},
        steps=[
            ExperienceRuleStep(
                step_id="step-1",
                source_transition_seqs=[1],
                before=RuleScreen(
                    url_template="wanted.co.kr/",
                    page_role="home",
                    reference_signature={"phash": "1" * 16, "size": [1000, 800]},
                ),
                actions=[
                    RuleAction(
                        source_seq=1,
                        action="click_marker",
                        target=RuleTarget(
                            description="검색 버튼",
                            reference={
                                "text": "검색",
                                "marker_type": "text",
                                "bbox_ratio": [0.7, 0.1, 0.9, 0.2],
                                "center_ratio": [0.8, 0.15],
                            },
                            reference_roi_signature={
                                "phash": "a" * 16,
                                "crop_rect_ratio": [0.65, 0.05, 0.95, 0.25],
                            },
                        ),
                    )
                ],
                intent=intent,
                applicable_when="홈 화면에 검색 버튼이 하나 보인다",
                decline_when="검색 버튼이 여러 개다",
                expected_effect=ExpectedEffect(
                    kind="page_change",
                    description="검색 화면이 열린다",
                    expected_url_template="wanted.co.kr/search",
                    expected_page_role="search",
                ),
            )
        ],
    )


def test_store_filters_rules_by_site_and_task_category(tmp_path):
    store = ExperienceRuleStore(tmp_path / "rules.db")
    store.save_rule(_rule(), source_run_id="run-1")

    assert len(store.get_site_rules("wanted", task_category="검색")) == 1
    assert store.get_site_rules("wanted", task_category="로그인") == []
    assert store.get_site_rules("saramin", task_category="검색") == []


def test_same_rule_counts_distinct_source_runs_once(tmp_path):
    store = ExperienceRuleStore(tmp_path / "support.db")
    store.save_rule(_rule(), source_run_id="run-1")
    store.save_rule(_rule(), source_run_id="run-1")
    store.save_rule(_rule(), source_run_id="run-2")

    payload = store.get_by_site("wanted")[0]
    assert payload["support_count"] == 2
    assert payload["source_count"] == 2


def test_changed_rule_replaces_same_purpose_and_resets_runtime_stats(tmp_path):
    store = ExperienceRuleStore(tmp_path / "replace.db")
    store.save_rule(_rule(), source_run_id="run-1")
    rule_key = store.get_site_rules("wanted")[0][0]
    store.record_replay_result(rule_key, True)
    store.save_rule(_rule(intent="검색 패널을 연다"), source_run_id="run-2")

    payload = store.get_by_site("wanted")[0]
    assert payload["steps"][0]["intent"] == "검색 패널을 연다"
    assert payload["support_count"] == 1
    assert payload["replay_success_count"] == 0


def test_new_store_removes_legacy_active_recipe_tables(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE recipes (recipe_key TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE recipe_sources (recipe_key TEXT, run_id TEXT)")

    ExperienceRuleStore(db_path)

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "recipes" not in tables
    assert "recipe_sources" not in tables
    assert "experience_rules" in tables
