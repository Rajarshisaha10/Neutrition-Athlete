from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.database.repository import DiagnosisRepository
from src.diagnosis_model.hybrid import DiagnosisEngine
from src.feedback.service import FeedbackService
from src.inference import InjuryRiskPredictor
from src.questionnaire.schemas import ConfirmationRequest, DiagnosisRequest, FeedbackRequest


app = FastAPI(
    title="Athlete Injury Risk Prediction API",
    version="1.0.0",
    description="Predicts calibrated athlete injury risk probability. Not a diagnostic system.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "app" / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

_predictor: InjuryRiskPredictor | None = None
_diagnosis_engine: DiagnosisEngine | None = None
_diagnosis_repository: DiagnosisRepository | None = None


class PredictionRequest(BaseModel):
    metrics: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


def predictor() -> InjuryRiskPredictor:
    global _predictor
    if _predictor is None:
        _predictor = InjuryRiskPredictor()
    return _predictor


def diagnosis_engine() -> DiagnosisEngine:
    global _diagnosis_engine
    if _diagnosis_engine is None:
        _diagnosis_engine = DiagnosisEngine()
    return _diagnosis_engine


def diagnosis_repository() -> DiagnosisRepository:
    global _diagnosis_repository
    if _diagnosis_repository is None:
        _diagnosis_repository = DiagnosisRepository()
    return _diagnosis_repository


@app.get("/health")
def health():
    try:
        p = predictor()
        return {
            "status": "ok",
            "selected_model": p.artifact.get("selected_model"),
            "threshold": p.threshold,
            "diagnosis_engine": "ready",
        }
    except Exception as exc:
        return {"status": "not_ready", "detail": str(exc)}


@app.get("/")
def frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"status": "ok", "detail": "Frontend assets not found."}


@app.post("/predict")
def predict(payload: Dict[str, Any]):
    try:
        return predictor().predict_one(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/batch_predict")
def batch_predict(payload: List[Dict[str, Any]]):
    try:
        return {"predictions": predictor().predict_batch(payload)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/diagnose")
def diagnose(payload: DiagnosisRequest):
    try:
        response = diagnosis_engine().diagnose(payload)
        try:
            diagnosis_repository().store_assessment(payload, response)
        except Exception:
            pass
        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/confirm-diagnosis")
def confirm_diagnosis(payload: ConfirmationRequest):
    try:
        return FeedbackService(diagnosis_repository()).confirm_diagnosis(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    try:
        return FeedbackService(diagnosis_repository()).store_feedback(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/assessment/{assessment_id}")
def get_assessment(assessment_id: str):
    try:
        assessment = diagnosis_repository().get_assessment(assessment_id)
        if assessment is None:
            raise HTTPException(status_code=404, detail="Assessment not found")
        return assessment
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
