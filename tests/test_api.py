from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_returns_200() -> None:
    response = client.get("/")

    assert response.status_code == 200


def test_status_contains_source_and_runtime() -> None:
    response = client.get("/api/status")

    assert response.status_code == 200
    data = response.json()
    assert "source" in data
    assert "runtime" in data
    assert data["source"]["source_mode"] in {"enetfall", "csv"}


def test_data_source_status_returns_200() -> None:
    response = client.get("/api/data-source/status")

    assert response.status_code == 200


def test_model_status_returns_200() -> None:
    response = client.get("/api/model/status")

    assert response.status_code == 200
    assert response.json()["model_name"] == "efficientnet_b0_enetfall"


def test_switch_detector_mode_simple_returns_200() -> None:
    response = client.post("/api/detector/mode", json={"mode": "simple"})

    assert response.status_code == 200
    assert response.json()["mode"] == "simple"


def test_switch_to_missing_csv_returns_400() -> None:
    response = client.post(
        "/api/data-source/csv",
        json={
            "csv_path": "data/not_exists.csv",
            "room": "lab",
            "device_id": "csv-node",
            "label": "unknown",
        },
    )

    assert response.status_code == 400


def test_recent_results_returns_200() -> None:
    response = client.get("/api/results/recent?limit=10")

    assert response.status_code == 200


def test_detector_reset_returns_200() -> None:
    response = client.post("/api/detector/reset")

    assert response.status_code == 200
