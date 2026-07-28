from agent.graph import worker_execution


def _action_request(tool_calls):
    from agent.graph.action_request import build_action_request

    return build_action_request("llm", "", tool_calls)


def test_sensitive_action_requires_user_approval_before_physical_input(monkeypatch):
    monkeypatch.setattr(
        worker_execution,
        "_dispatch_ui",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("민감 행동을 물리 입력으로 보내면 안 됩니다.")
        ),
    )

    result = worker_execution.action_node(
        {
            "current_markers": [
                {"id": 7, "text": "가입 신청", "bbox": [0, 0, 100, 20]},
            ],
            "current_url": "https://bank.example/product",
            "current_url_stale": False,
            "extracted_jd": {},
            "is_finished": False,
            "collected_data": [],
            "error_count": 0,
            "pending_action": _action_request(
                [
                    {
                        "name": "click_marker",
                        "args": {
                            "marker_id": 7,
                            "target_label": "가입 신청",
                            "risk_level": "sensitive",
                            "needs_user_confirmation": True,
                        },
                        "id": "sensitive-click",
                    }
                ]
            ),
        }
    )

    assert result["pending_human_approval"] is True
    assert result["human_approval_request"]["action"] == "click_marker"
    assert result["action_history"][0]["status"] == "skipped"


def test_local_api_rejects_untrusted_host_and_cross_origin_request():
    from fastapi.testclient import TestClient

    from agent.web_server import app

    client = TestClient(app)
    blocked_host = client.get(
        "/api/runs/missing",
        headers={"Host": "evil.example"},
    )
    preflight = client.options(
        "/api/chat",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert blocked_host.status_code == 400
    assert preflight.status_code == 400
    assert "access-control-allow-origin" not in preflight.headers
