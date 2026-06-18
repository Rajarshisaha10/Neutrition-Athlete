from __future__ import annotations

from src.database.repository import DiagnosisRepository
from src.questionnaire.schemas import ConfirmationRequest, FeedbackRequest


class FeedbackService:
    def __init__(self, repository: DiagnosisRepository | None = None):
        self.repository = repository or DiagnosisRepository()

    def confirm_diagnosis(self, payload: ConfirmationRequest) -> dict[str, str]:
        self.repository.confirm(payload)
        return {"status": "stored", "assessment_id": payload.assessment_id}

    def store_feedback(self, payload: FeedbackRequest) -> dict[str, str]:
        self.repository.feedback(payload)
        return {"status": "stored", "assessment_id": payload.assessment_id}
