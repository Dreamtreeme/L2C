import json

from benchmark.audit_e2e_history import build_history_audit, render_markdown


def _write_summary(
    path,
    *,
    commit="abc123",
    config="cfg-1",
    scenario="wanted-ios-warm",
    dirty=False,
    passed=True,
    duration=10.0,
):
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "git_commit": commit,
                "git_dirty": dirty,
                "config_fingerprint": config,
                "scenario_id": scenario,
                "site": "wanted",
                "run_mode": "warm",
                "query": "iOS 개발자",
                "target_count": 2,
                "status": "completed",
                "execution_time_sec": duration,
                "quality": {"passed": passed},
                "observability": {
                    "total_tokens": 100,
                    "estimated_cost_usd": 0.01,
                    "reflex_hits": 2,
                    "ocr_timeout_count": 0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_history_audit_separates_clean_and_dirty_runs(tmp_path):
    _write_summary(tmp_path / "clean.summary.json")
    _write_summary(tmp_path / "dirty.summary.json", dirty=True)

    audit = build_history_audit(tmp_path)

    assert audit["collection_run_count"] == 2
    assert audit["classification_counts"] == {
        "development": 1,
        "release": 1,
    }
    assert len(audit["groups"]) == 2


def test_history_audit_groups_only_matching_run_identity(tmp_path):
    _write_summary(tmp_path / "first.summary.json", duration=9.0)
    _write_summary(tmp_path / "second.summary.json", duration=11.0)
    _write_summary(
        tmp_path / "other-config.summary.json",
        config="cfg-2",
        duration=30.0,
    )

    audit = build_history_audit(tmp_path)
    repeated = next(group for group in audit["groups"] if group["count"] == 2)

    assert audit["repeated_group_count"] == 1
    assert repeated["passed_count"] == 2
    assert repeated["execution_time_sec"] == {
        "min": 9.0,
        "median": 10.0,
        "max": 11.0,
    }


def test_history_audit_markdown_uses_raw_range_without_p95(tmp_path):
    _write_summary(tmp_path / "first.summary.json", duration=9.0)
    _write_summary(
        tmp_path / "second.summary.json",
        duration=11.0,
        passed=False,
    )
    (tmp_path / "matrix.summary.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )

    markdown = render_markdown(build_history_audit(tmp_path))

    assert "1/2" in markdown
    assert "9.00/10.00/11.00" in markdown
    assert "| p95 |" not in markdown
    assert "작은 표본에는 p95를 계산하지 않고" in markdown
    assert "수집 E2E가 아닌 summary: `1`개" in markdown
