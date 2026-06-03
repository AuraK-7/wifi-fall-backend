from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_mobile_fall_event_can_be_replayed() -> None:
    event_id = "mobile-fall-test-pkt"
    payload = {
        "event_id": event_id,
        "packet_id": "pkt-test",
        "sequence_id": "seq-test",
        "timestamp": 1710000000.123,
        "room": "demo_room",
        "device_id": "mobile-detector-test",
        "model": {
            "runtime": "mock",
            "weight_url": "/models/mobile-fall.onnx",
            "input_shape": [1, 64, 30],
            "class_names": ["non_fall", "fall"],
            "threshold": 0.75,
        },
        "packet": {
            "packet_id": "pkt-test",
            "sequence_id": "seq-test",
            "frame_id": 1024,
            "timestamp": 1710000000.123,
            "room": "demo_room",
            "device_id": "console-csi-001",
            "source": "console",
            "mode": "stream",
            "subcarrier_count": 30,
            "window_size": 2,
            "subcarriers": [0.1, 0.2, 0.3],
            "window": [
                {
                    "frame_index": 0,
                    "timestamp": 1710000000.123,
                    "subcarriers": [0.1, 0.2, 0.3],
                    "energy": 0.31,
                    "variance": 0.02,
                },
                {
                    "frame_index": 1,
                    "timestamp": 1710000000.223,
                    "subcarriers": [0.6, 0.8, 0.7],
                    "energy": 0.72,
                    "variance": 0.13,
                },
            ],
        },
        "result": {
            "predicted_label": "fall",
            "confidence": 0.91,
            "risk_level": "high",
            "alert": True,
            "activity_score": 0.88,
            "energy": 0.72,
            "variance": 0.13,
            "reason": "mobile mock fall",
            "avatar": {
                "display_state": "fallen",
                "dataset_state": "unknown",
                "predicted_state": "fallen",
                "source": "mobile_model",
                "dataset_label": None,
                "predicted_label": "fall",
                "confidence": 0.91,
                "risk_level": "high",
                "alert": True,
            },
        },
        "analytics": {
            "micro_doppler_spectrum": [0.1, 0.2],
            "subcarrier_amplitudes": [0.1, 0.2, 0.3],
            "antenna_correlation": 0.8,
            "energy": 0.72,
            "dominant_freq": 3.1,
            "frequency_spread": 5.2,
            "signal_variance": 0.13,
        },
    }

    save_response = client.post("/api/mobile/fall-events", json=payload)
    assert save_response.status_code == 200
    assert save_response.json()["saved"] is True

    replay_response = client.get(f"/api/events/{event_id}/replay")
    assert replay_response.status_code == 200
    replay = replay_response.json()
    assert replay["window_count"] == 2
    assert replay["windows"][1]["avatar"]["display_state"] == "fallen"


def test_demo_packet_rejects_ground_truth_label() -> None:
    response = client.post(
        "/api/demo/packets",
        json={
            "packet_id": "pkt-with-label",
            "sequence_id": "seq-test",
            "frame_id": 1,
            "timestamp": 1710000000.123,
            "room": "demo_room",
            "device_id": "console-csi-001",
            "source": "console",
            "mode": "single",
            "subcarrier_count": 30,
            "window_size": 1,
            "subcarriers": [0.1, 0.2, 0.3],
            "window": [],
            "label": "fall",
        },
    )

    assert response.status_code == 422
