from fastapi.testclient import TestClient

from app.db.database import Base, SessionLocal, engine
from app.main import app
from app.schemas.alert import AlertEventCreate, AlertEventUpdate
from app.services.alert import AlertService

Base.metadata.create_all(bind=engine)

client = TestClient(app)
alert_service = AlertService()


def _alert_payload() -> AlertEventCreate:
    return AlertEventCreate(
        timestamp=1234567890.0,
        room="bedroom",
        device_id="sim-node-test",
        predicted_label="fall",
        confidence=0.91,
        risk_level="high",
        activity_score=0.88,
        reason="test fall alert",
    )


def test_alert_list_api() -> None:
    response = client.get("/api/alerts")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_alert_summary_api() -> None:
    response = client.get("/api/alerts/summary/count")

    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "handled" in data
    assert "unhandled" in data


def test_alert_service_create_and_get() -> None:
    db = SessionLocal()
    try:
        created = alert_service.create_alert(db, _alert_payload())
        found = alert_service.get_alert(db, created.event_id)

        assert found is not None
        assert found.event_id == created.event_id
    finally:
        db.close()


def test_alert_service_update() -> None:
    db = SessionLocal()
    try:
        created = alert_service.create_alert(db, _alert_payload())
        updated = alert_service.update_alert(
            db,
            created.event_id,
            AlertEventUpdate(handled=True, handler_note="confirmed safe"),
        )

        assert updated is not None
        assert updated.handled is True
        assert updated.handler_note == "confirmed safe"
    finally:
        db.close()


def test_alert_detail_and_patch_api() -> None:
    db = SessionLocal()
    try:
        created = alert_service.create_alert(db, _alert_payload())
        event_id = created.event_id
    finally:
        db.close()

    detail_response = client.get(f"/api/alerts/{event_id}")

    assert detail_response.status_code == 200

    patch_response = client.patch(
        f"/api/alerts/{event_id}",
        json={
            "handled": True,
            "handler_note": "已确认老人安全",
        },
    )

    assert patch_response.status_code == 200
    assert patch_response.json()["handled"] is True
