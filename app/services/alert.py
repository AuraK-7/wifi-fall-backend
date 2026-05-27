import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import AlertEvent
from app.schemas.alert import AlertEventCreate, AlertEventUpdate


class AlertService:
    def create_alert(self, db: Session, alert_in: AlertEventCreate) -> AlertEvent:
        alert = AlertEvent(
            event_id=str(uuid.uuid4()),
            timestamp=alert_in.timestamp,
            room=alert_in.room,
            device_id=alert_in.device_id,
            predicted_label=alert_in.predicted_label,
            confidence=alert_in.confidence,
            risk_level=alert_in.risk_level,
            activity_score=alert_in.activity_score,
            reason=alert_in.reason,
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)
        return alert

    def list_alerts(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 50,
        handled: bool | None = None,
    ) -> list[AlertEvent]:
        statement = select(AlertEvent)
        if handled is not None:
            statement = statement.where(AlertEvent.handled == handled)

        statement = statement.order_by(AlertEvent.created_at.desc()).offset(skip).limit(limit)
        return list(db.scalars(statement).all())

    def get_alert(self, db: Session, event_id: str) -> AlertEvent | None:
        statement = select(AlertEvent).where(AlertEvent.event_id == event_id)
        return db.scalars(statement).first()

    def update_alert(
        self,
        db: Session,
        event_id: str,
        update_in: AlertEventUpdate,
    ) -> AlertEvent | None:
        alert = self.get_alert(db, event_id)
        if alert is None:
            return None

        if update_in.handled is not None:
            alert.handled = update_in.handled
        if update_in.handler_note is not None:
            alert.handler_note = update_in.handler_note

        db.commit()
        db.refresh(alert)
        return alert

    def count_alerts(self, db: Session, handled: bool | None = None) -> int:
        statement = select(func.count()).select_from(AlertEvent)
        if handled is not None:
            statement = statement.where(AlertEvent.handled == handled)

        return int(db.scalar(statement) or 0)
