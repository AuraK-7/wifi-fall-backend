from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_returns_200() -> None:
    response = client.get("/")

    assert response.status_code == 200


def test_status_contains_simulator_and_runtime() -> None:
    response = client.get("/api/status")

    assert response.status_code == 200
    data = response.json()
    assert "simulator" in data
    assert "runtime" in data


def test_update_label_walking_returns_200() -> None:
    response = client.post("/api/simulator/label/walking")

    assert response.status_code == 200


def test_update_label_fall_returns_200() -> None:
    response = client.post("/api/simulator/label/fall")

    assert response.status_code == 200


def test_update_label_bad_label_returns_400() -> None:
    response = client.post("/api/simulator/label/bad_label")

    assert response.status_code == 400


def test_update_room_returns_200() -> None:
    response = client.post("/api/simulator/room/living_room")

    assert response.status_code == 200


def test_update_device_returns_200() -> None:
    response = client.post("/api/simulator/device/sim-node-002")

    assert response.status_code == 200


def test_load_sequence_returns_200() -> None:
    sequence = [
        {"label": "walking", "duration_frames": 50},
        {"label": "sitting", "duration_frames": 30},
        {"label": "fall", "duration_frames": 60},
        {"label": "lying", "duration_frames": 80},
    ]

    response = client.post("/api/simulator/sequence", json=sequence)

    assert response.status_code == 200


def test_clear_sequence_returns_200() -> None:
    response = client.delete("/api/simulator/sequence")

    assert response.status_code == 200


def test_recent_results_returns_200() -> None:
    response = client.get("/api/results/recent?limit=10")

    assert response.status_code == 200


def test_detector_reset_returns_200() -> None:
    response = client.post("/api/detector/reset")

    assert response.status_code == 200
