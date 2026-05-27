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


def test_update_simulator_label_returns_success() -> None:
    response = client.post("/api/simulator/label/fall")

    assert response.status_code == 200


def test_update_simulator_label_rejects_invalid_label() -> None:
    response = client.post("/api/simulator/label/invalid_label")

    assert response.status_code == 400
