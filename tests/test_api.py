from fastapi.testclient import TestClient

from app.main import _avatar_payload, app
from app.schemas.csi import CsiFrame, DetectionResult


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


def test_avatar_payload_maps_prediction_to_3d_state() -> None:
    frame = CsiFrame(
        frame_id=1,
        device_id="test-node",
        timestamp=123.0,
        room="lab",
        subcarriers=[0.1],
        simulated_label="non_fall",
        label="non_fall",
    )
    result = DetectionResult(
        timestamp=123.0,
        room="lab",
        predicted_label="fall",
        confidence=0.91,
        risk_level="high",
        alert=True,
        activity_score=0.91,
    )

    avatar = _avatar_payload(frame, result)

    assert avatar["display_state"] == "standing"
    assert avatar["dataset_state"] == "standing"
    assert avatar["predicted_state"] == "fallen"
    assert avatar["source"] == "dataset_label"
