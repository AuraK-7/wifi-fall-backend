from app.schemas.csi import DetectionResult


class AlertService:
    def should_alert(self, result: DetectionResult) -> bool:
        return result.alert
