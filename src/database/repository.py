from __future__ import annotations

import json
from typing import Any, Optional

from src.database.models import ConfirmedDiagnosis, InjuryAssessment, ModelFeedback, ReviewQueue, User
from src.database.session import init_db, session_scope
from src.questionnaire.schemas import ConfirmationRequest, DiagnosisRequest, DiagnosisResponse, FeedbackRequest


class DiagnosisRepository:
    def __init__(self) -> None:
        init_db()

    def store_assessment(self, request: DiagnosisRequest, response: DiagnosisResponse) -> None:
        with session_scope() as session:
            if request.user_id and session.get(User, request.user_id) is None:
                session.add(User(id=request.user_id))
            row = InjuryAssessment(
                id=response.assessment_id,
                user_id=request.user_id,
                questionnaire_json=request.model_dump_json(),
                risk_model_output_json=request.risk_model_output.model_dump_json() if request.risk_model_output else None,
                prediction_json=response.model_dump_json(),
                confidence=response.confidence,
                model_version=response.model_version,
                insufficient_confidence=response.insufficient_confidence,
            )
            session.merge(row)
            if response.insufficient_confidence:
                session.add(ReviewQueue(assessment_id=response.assessment_id, reason="low_confidence", priority=10))

    def confirm(self, payload: ConfirmationRequest) -> None:
        with session_scope() as session:
            session.add(
                ConfirmedDiagnosis(
                    assessment_id=payload.assessment_id,
                    confirmed_diagnosis=payload.confirmed_diagnosis,
                    clinician_type=payload.clinician_type,
                    notes=payload.notes,
                )
            )

    def feedback(self, payload: FeedbackRequest) -> None:
        with session_scope() as session:
            session.add(
                ModelFeedback(
                    assessment_id=payload.assessment_id,
                    rating=payload.rating,
                    helpful=payload.helpful,
                    comment=payload.comment,
                )
            )

    def get_assessment(self, assessment_id: str) -> Optional[dict[str, Any]]:
        with session_scope() as session:
            row = session.get(InjuryAssessment, assessment_id)
            if row is None:
                return None
            confirmation = row.confirmation
            return {
                "assessment_id": row.id,
                "questionnaire_data": json.loads(row.questionnaire_json),
                "prediction": json.loads(row.prediction_json),
                "confidence": row.confidence,
                "model_version": row.model_version,
                "confirmed_diagnosis": confirmation.confirmed_diagnosis if confirmation else None,
                "timestamp": row.created_at.isoformat(),
            }

    def confirmed_cases(self) -> list[dict[str, Any]]:
        with session_scope() as session:
            rows = (
                session.query(InjuryAssessment, ConfirmedDiagnosis)
                .join(ConfirmedDiagnosis, ConfirmedDiagnosis.assessment_id == InjuryAssessment.id)
                .all()
            )
            return [
                {
                    "questionnaire_data": json.loads(assessment.questionnaire_json),
                    "prediction": json.loads(assessment.prediction_json),
                    "confirmed_diagnosis": confirmation.confirmed_diagnosis,
                    "confidence": assessment.confidence,
                    "timestamp": confirmation.created_at.isoformat(),
                }
                for assessment, confirmation in rows
            ]
