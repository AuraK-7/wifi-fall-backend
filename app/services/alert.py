from app.schemas.csi import FallDetectionResult


class AlertService:
    def should_alert(self, result: FallDetectionResult) -> bool:
        return result.status == "fall_suspected"
