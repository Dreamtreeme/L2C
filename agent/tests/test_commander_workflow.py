import json


def _worker_result(
    site="wanted",
    attempt=0,
    count=1,
    run_id="worker-run-1",
    hit_recursion_limit=False,
    is_finished=True,
    recursion_limit=60,
    final_state=None,
    target_count=0,
):
    submission = {
        "run_id": run_id,
        "goal": "collect AI engineer jobs",
        "site": site,
        "keyword": "AI engineer trend",
        "run_status": "recursion_limit" if hit_recursion_limit and not is_finished else "finished",
        "review_attempt": attempt,
        "collected_count": count,
        "target_count": target_count,
        "persisted_count": 0,
        "recorded_steps": [
            {"seq": 0, "state_key": "state-a", "action": "click_marker", "target": {"text": "AI Engineer"}}
        ] if count else [],
        "feedback_episodes": [
            {"seq": 0, "proposal": {"action": "click_marker"}, "feedback": {"label": "partial"}}
        ] if count else [],
        "extracted_summary": {
            "has_data": bool(count),
            "job_count": count,
            "current_url": "https://www.wanted.co.kr/wd/1" if count else "",
            "jobs": [
                {
                    "company": "Acme",
                    "position": "AI Engineer",
                    "url": "https://example.com/jobs/1",
                    "field_count": 5,
                }
            ] if count else [],
        },
    }
    extracted = {
        "jobs": [
            {"company_name": "Acme", "position": "AI Engineer", "url": "https://example.com/jobs/1"}
        ]
    } if count else {}
    final_state = final_state or {
        "current_url": "https://www.wanted.co.kr/wd/1" if count else "",
        "plan": [
            "search jobs",
            "open first card",
            "collect detail",
            "go back",
            "open next card",
        ],
        "current_plan_step": 3,
    }
    return {
        "submission": submission,
        "extracted_jd": extracted,
        "final_state": final_state,
        "site_slug": site,
        "site_name": site,
        "keyword": "AI engineer trend",
        "target_count": target_count,
        "hit_recursion_limit": hit_recursion_limit,
        "is_finished": is_finished,
        "recursion_limit": recursion_limit,
    }


def test_select_sites_prefers_explicit_site():
    from agent.graph.commander_workflow import _select_sites_for_query

    sites = [
        {"slug": "wanted", "display_name": "Wanted", "domains": ["wanted.co.kr"]},
        {"slug": "saramin", "display_name": "Saramin", "domains": ["saramin.co.kr"]},
    ]

    assert _select_sites_for_query("wanted AI engineer", sites) == ["wanted"]
    assert _select_sites_for_query("AI engineer trend", sites) == ["wanted", "saramin"]


def test_commander_graph_retries_revised_worker(monkeypatch):
    import agent.graph.commander_workflow as cw

    calls = []

    def fake_run_worker_once(query, site=None, review_feedback=None, review_attempt=0, run_id=None):
        calls.append({"query": query, "site": site, "feedback": review_feedback, "attempt": review_attempt, "run_id": run_id})
        return _worker_result(site=site, attempt=review_attempt, count=0 if review_attempt == 0 else 1)

    def fake_commit_worker_review(submission, source=""):
        if submission.get("collected_count", 0) <= 0:
            return {
                "decision": "revise",
                "reasons": ["missing data"],
                "feedback_to_worker": "collect at least one valid job",
                "recipe_candidate": False,
                "confidence": 0.8,
            }, f"{submission['run_id']}:{submission['review_attempt']}"
        return {
            "decision": "accept",
            "reasons": ["ok"],
            "feedback_to_worker": "",
            "recipe_candidate": True,
            "confidence": 0.7,
        }, f"{submission['run_id']}:{submission['review_attempt']}"

    def fake_persist(worker_result, review, source=""):
        submission = dict(worker_result["submission"])
        submission["persisted_count"] = 1
        return 1, submission, review, f"{submission['run_id']}:{submission['review_attempt']}"

    monkeypatch.setattr(cw, "run_worker_once", fake_run_worker_once)
    monkeypatch.setattr(cw, "commit_worker_review", fake_commit_worker_review)
    monkeypatch.setattr(cw, "persist_accepted_worker_result", fake_persist)
    monkeypatch.setattr(cw, "_close_browser_after_run", lambda: None)
    monkeypatch.setattr(cw, "query_recent_jobs", lambda: "<document id=\"1\">Acme AI Engineer</document>")

    result = cw.run_commander_graph("AI engineer trend", site_queue=["wanted"])

    assert [call["attempt"] for call in calls] == [0, 1]
    assert calls[1]["feedback"] and "collect at least one" in calls[1]["feedback"]
    assert result["accepted_sites"] == ["wanted"]
    assert result.get("failed_sites", []) == []
    assert "accepted_sites=1" in result["final_answer"]


def test_commander_graph_marks_failed_when_retry_budget_exhausted(monkeypatch):
    import agent.graph.commander_workflow as cw

    def fake_run_worker_once(query, site=None, review_feedback=None, review_attempt=0, run_id=None):
        return _worker_result(site=site, attempt=review_attempt, count=0)

    def fake_commit_worker_review(submission, source=""):
        return {
            "decision": "revise",
            "reasons": ["missing data"],
            "feedback_to_worker": "collect data",
            "recipe_candidate": False,
            "confidence": 0.8,
        }, f"{submission['run_id']}:{submission['review_attempt']}"

    monkeypatch.setattr(cw, "run_worker_once", fake_run_worker_once)
    monkeypatch.setattr(cw, "commit_worker_review", fake_commit_worker_review)
    monkeypatch.setattr(cw, "_close_browser_after_run", lambda: None)
    monkeypatch.setattr(cw, "query_recent_jobs", lambda: "no rows")

    app = cw.build_commander_graph()
    result = app.invoke(
        {
            "user_query": "AI engineer trend",
            "site_queue": ["wanted"],
            "current_site_index": 0,
            "max_review_retries": 0,
            "worker_submissions": [],
            "reviews": [],
            "accepted_sites": [],
            "failed_sites": [],
        }
    )

    assert result.get("accepted_sites", []) == []
    assert len(result["failed_sites"]) == 1
    assert result["failed_sites"][0]["site"] == "wanted"
    assert "failed_sites=1" in result["final_answer"]


def test_commander_graph_reports_before_raising_recursion_limit(monkeypatch):
    import agent.graph.commander_workflow as cw

    calls = []

    def fake_run_worker_once(query, site=None, review_feedback=None, review_attempt=0, run_id=None):
        calls.append(site)
        return _worker_result(
            site=site,
            count=4,
            hit_recursion_limit=True,
            is_finished=False,
            recursion_limit=60,
        )

    def fake_commit_worker_review(submission, source=""):
        return {
            "decision": "accept",
            "reasons": ["partial data is valid"],
            "feedback_to_worker": "",
            "recipe_candidate": True,
            "confidence": 0.8,
        }, submission["run_id"]

    def fake_persist(worker_result, review, source=""):
        submission = dict(worker_result["submission"])
        submission["persisted_count"] = 4
        return 4, submission, review, submission["run_id"]

    monkeypatch.setenv("VISION_HITL_ON_RECURSION_LIMIT", "1")
    monkeypatch.delenv("VISION_AGENT_RECURSION_LIMIT_INCREMENT", raising=False)
    monkeypatch.setattr(cw, "run_worker_once", fake_run_worker_once)
    monkeypatch.setattr(cw, "commit_worker_review", fake_commit_worker_review)
    monkeypatch.setattr(cw, "persist_accepted_worker_result", fake_persist)
    monkeypatch.setattr(cw, "_close_browser_after_run", lambda: None)
    monkeypatch.setattr(cw, "query_recent_jobs", lambda: "<document id=\"1\">Acme AI Engineer</document>")

    result = cw.run_commander_graph("AI engineer trend", site_queue=["wanted", "saramin"])

    assert calls == ["wanted"]
    assert result["pending_human_approval"] is True
    assert result["current_site_index"] == 0
    assert result["intermediate_report"]["current_recursion_limit"] == 60
    assert result["intermediate_report"]["suggested_recursion_limit"] == 120
    assert "중간보고" in result["final_answer"]
    assert "120" in result["final_answer"]
    assert "계속" in result["final_answer"]


def test_commander_graph_does_not_request_more_limit_after_collection_target(monkeypatch):
    import agent.graph.commander_workflow as cw

    calls = []

    def fake_run_worker_once(query, site=None, review_feedback=None, review_attempt=0, run_id=None):
        calls.append(site)
        return _worker_result(
            site=site,
            count=8,
            target_count=8,
            hit_recursion_limit=True,
            is_finished=False,
            recursion_limit=120,
            final_state={
                "current_url": "https://www.wanted.co.kr/search?query=iOS",
                "plan": [
                    "open search results",
                    "open first card and collect detail",
                    "open eighth card and collect detail",
                    "go_back",
                    "submit collected data",
                ],
                "current_plan_step": 3,
            },
        )

    def fake_commit_worker_review(submission, source=""):
        return {
            "decision": "accept",
            "reasons": ["target data collected"],
            "feedback_to_worker": "",
            "recipe_candidate": True,
            "confidence": 0.8,
        }, submission["run_id"]

    def fake_persist(worker_result, review, source=""):
        submission = dict(worker_result["submission"])
        submission["persisted_count"] = 8
        return 8, submission, review, submission["run_id"]

    monkeypatch.setenv("VISION_HITL_ON_RECURSION_LIMIT", "1")
    monkeypatch.setattr(cw, "run_worker_once", fake_run_worker_once)
    monkeypatch.setattr(cw, "commit_worker_review", fake_commit_worker_review)
    monkeypatch.setattr(cw, "persist_accepted_worker_result", fake_persist)
    monkeypatch.setattr(cw, "_close_browser_after_run", lambda: None)
    monkeypatch.setattr(cw, "query_recent_jobs", lambda: "<document id=\"1\">8 iOS jobs</document>")

    result = cw.run_commander_graph("iOS developer jobs", site_queue=["wanted"])

    assert calls == ["wanted"]
    assert result["pending_human_approval"] is False
    assert result["current_site_index"] == 1
    assert "중간보고" not in result["final_answer"]
    assert "accepted_sites=1" in result["final_answer"]


def test_qa_reasoning_node_can_route_to_commander_graph(monkeypatch):
    from agent.graph.nodes import qa_reasoning_node

    monkeypatch.setenv("COMMANDER_GRAPH_ENABLED", "1")
    monkeypatch.setattr(
        "agent.graph.commander_workflow.run_commander_graph",
        lambda query: {"final_answer": f"graph answer for {query}"},
    )

    result = qa_reasoning_node({"goal": "AI engineer trend"})

    assert result["last_action_result"] == "graph answer for AI engineer trend"
    assert result["is_finished"] is True
    assert result["step_durations"][0]["node"] == "qa_commander_graph"
