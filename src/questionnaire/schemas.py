from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class AthleteProfile(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    sport: Optional[str] = None
    competition_level: Optional[str] = None


class Symptoms(BaseModel):
    pain_severity: Optional[float] = Field(default=None, ge=0, le=10)
    swelling: bool = False
    bruising: bool = False
    instability: bool = False
    locking: bool = False
    clicking: bool = False
    popping_sound: bool = False
    reduced_range_of_motion: bool = False
    weight_bearing_ability: Optional[str] = None
    localized_tenderness: bool = False
    weakness: bool = False
    dizziness: bool = False
    headache: bool = False
    confusion: bool = False
    nausea: bool = False
    loss_of_consciousness: bool = False
    morning_stiffness: bool = False
    night_pain: bool = False
    pain_overhead: bool = False
    grip_pain: bool = False
    pain_with_jump: bool = False
    pain_with_deep_bend: bool = False
    medial_pain: bool = False
    worsening_headache: bool = False
    repeated_vomiting: bool = False


class Mechanism(BaseModel):
    twisting: bool = False
    overuse: bool = False
    direct_impact: bool = False
    landing: bool = False
    sprinting: bool = False
    running: bool = False
    throwing: bool = False
    jumping: bool = False
    cutting: bool = False
    fall: bool = False
    kicking: bool = False
    squatting: bool = False


class RiskModelOutput(BaseModel):
    injury_risk_probability: Optional[float] = Field(default=None, ge=0, le=1)
    fatigue_score: Optional[float] = None
    workload_spike: Optional[float] = None
    sleep_quality: Optional[float] = None
    recovery_index: Optional[float] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class DiagnosisRequest(BaseModel):
    assessment_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    athlete_profile: AthleteProfile = Field(default_factory=AthleteProfile)
    pain_location: str
    onset: Optional[str] = None
    symptoms: Symptoms = Field(default_factory=Symptoms)
    mechanism: Mechanism = Field(default_factory=Mechanism)
    risk_model_output: Optional[RiskModelOutput] = None
    athlete_metrics: Optional[Dict[str, Any]] = None


class InjuryPrediction(BaseModel):
    injury: str
    probability: float
    symptom_match_score: float = 0.0
    candidate_status: str = "candidate"


class DiagnosisResponse(BaseModel):
    assessment_id: str
    top_predictions: List[InjuryPrediction]
    confidence: float
    explanation: List[str]
    reasoning: List[str]
    insufficient_confidence: bool = False
    message: Optional[str] = None
    risk_model_used: bool = False
    model_version: str = "knowledge-v1"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConfirmationRequest(BaseModel):
    assessment_id: str
    confirmed_diagnosis: str
    clinician_type: Optional[str] = None
    notes: Optional[str] = None


class FeedbackRequest(BaseModel):
    assessment_id: str
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    helpful: Optional[bool] = None
    comment: Optional[str] = None
