from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services.data_source_manager import DataSourceManager


client = TestClient(app)


def test_data_source_status_api_returns_200() -> None:
    response = client.get("/api/data-source/status")

    assert response.status_code == 200
    assert "source_mode" in response.json()


def test_switch_to_csv_api_returns_200(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample_csi.csv"
    csv_path.write_text(
        "timestamp,label,sc0,sc1,sc2\n"
        "1.0,walking,0.1,0.2,0.3\n"
        "2.0,fall,0.5,0.9,0.4\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/data-source/csv",
        json={
            "csv_path": str(csv_path),
            "room": "lab",
            "device_id": "csv-test-node",
            "label": "unknown",
        },
    )

    assert response.status_code == 200
    assert response.json()["source"]["source_mode"] == "csv"


def test_switch_to_missing_enetfall_source_returns_400() -> None:
    response = client.post(
        "/api/data-source/enetfall",
        json={
            "data_dir": "data/not_exists_enetfall",
            "dataset_names": ["missing.mat"],
            "room": "home",
            "device_id": "enetfall-test-node",
        },
    )

    assert response.status_code == 400


def test_csv_replay_source_returns_fixed_length_frame(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample_csi.csv"
    csv_path.write_text(
        "timestamp,label,sc0,sc1,sc2\n"
        "1.0,walking,0.1,0.2,0.3\n",
        encoding="utf-8",
    )
    manager = DataSourceManager()
    source = manager.switch_to_csv(
        csv_path=str(csv_path),
        room="lab",
        device_id="csv-test-node",
        label="unknown",
    )

    frame = source.next_frame()

    assert frame.device_id == "csv-test-node"
    assert frame.room == "lab"
    assert frame.simulated_label == "walking"
    assert len(frame.subcarriers) == settings.CSI_SUBCARRIER_COUNT
