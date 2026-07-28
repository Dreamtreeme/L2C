from pathlib import Path

from fastapi.testclient import TestClient


def test_root_serves_react_build_when_available(tmp_path, monkeypatch):
    from agent import web_server

    index_path = tmp_path / "index.html"
    index_path.write_text("<html><body>react-workspace</body></html>", encoding="utf-8")
    monkeypatch.setattr(web_server, "frontend_index_path", index_path)

    response = TestClient(web_server.app).get("/")

    assert response.status_code == 200
    assert "react-workspace" in response.text


def test_root_reports_missing_frontend_build(tmp_path, monkeypatch):
    from agent import web_server

    missing_index = Path(tmp_path) / "missing" / "index.html"
    monkeypatch.setattr(web_server, "frontend_index_path", missing_index)

    response = TestClient(web_server.app).get("/")

    assert response.status_code == 503
    assert "프론트엔드 빌드가 없습니다" in response.json()["detail"]
