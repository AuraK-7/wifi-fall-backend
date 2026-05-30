from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_returns_running_status() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_status_returns_current_system_state() -> None:
    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["source"]["source_mode"] in {"enetfall", "csv"}
