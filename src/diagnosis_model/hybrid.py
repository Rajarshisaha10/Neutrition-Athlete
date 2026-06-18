from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from src.candidate_filtering.filter import CandidateFilter
from src.inference import InjuryRiskPredictor
from src.knowledge_engine.knowledge_base import InjuryKnowledgeBase, InjuryProfile
from src.questionnaire.schemas import DiagnosisRequest, DiagnosisResponse, InjuryPrediction, RiskModelOutput
from src.symptom_matching.matcher import SymptomMatcher


LOW_CONFIDENCE_MESSAGE = "Insufficient confidence for reliable injury classification."


class DiagnosisEngine:
    def __init__(
        self,
        knowledge_base: Optional[InjuryKnowledgeBase] = None,
        diagnosis_model_path: str | Path = "models/diagnosis/latest.joblib",
    ):
        self.knowledge_base = knowledge_base or InjuryKnowledgeBase()
        self.filter = CandidateFilter(self.knowledge_base)
        self.matcher = SymptomMatcher()
        self.diagnosis_model_path = Path(diagnosis_model_path)
        self.ml_artifact = self._load_ml_artifact()
        self.risk_predictor: Optional[InjuryRiskPredictor] = None

    def _load_ml_artifact(self) -> Optional[Dict[str, Any]]:
        if not self.diagnosis_model_path.exists():
            return None
        return joblib.load(self.diagnosis_model_path)

    def diagnose(self, request: DiagnosisRequest) -> DiagnosisResponse:
        risk_output = self._risk_output(request)
        candidates = self.filter.filter(request)
        scored: List[Tuple[str, InjuryProfile, float, float, str, List[str]]] = []

        for name, (profile, candidate_multiplier, status) in candidates.items():
            match_score, reasoning = self.matcher.score(request, profile)
            rule_score = self._rule_probability_proxy(match_score, profile, risk_output) * candidate_multiplier
            scored.append((name, profile, match_score, rule_score, status, reasoning))

        if not scored:
            return DiagnosisResponse(
                assessment_id=request.assessment_id,
                top_predictions=[],
                confidence=0.0,
                explanation=["Pain location does not match any supported injury region."],
                reasoning=[],
                insufficient_confidence=True,
                message=LOW_CONFIDENCE_MESSAGE,
                risk_model_used=risk_output is not None,
                model_version=self.knowledge_base.version,
            )

        probabilities = self._combine_probabilities(request, scored)
        predictions = [
            InjuryPrediction(
                injury=name,
                probability=round(float(probabilities[name]), 4),
                symptom_match_score=round(float(match_score), 4),
                candidate_status=status,
            )
            for name, _, match_score, _, status, _ in scored
        ]
        predictions.sort(key=lambda item: item.probability, reverse=True)
        top_predictions = predictions[:3]
        confidence = self._confidence(top_predictions)
        top_name = top_predictions[0].injury if top_predictions else None
        explanation = self._explanation(request, top_name, scored, risk_output)
        insufficient = confidence < 0.55

        return DiagnosisResponse(
            assessment_id=request.assessment_id,
            top_predictions=top_predictions,
            confidence=round(float(confidence), 4),
            explanation=explanation,
            reasoning=explanation,
            insufficient_confidence=insufficient,
            message=LOW_CONFIDENCE_MESSAGE if insufficient else None,
            risk_model_used=risk_output is not None,
            model_version=self._model_version(),
            timestamp=datetime.utcnow(),
        )

    def _risk_output(self, request: DiagnosisRequest) -> Optional[RiskModelOutput]:
        if request.risk_model_output is not None:
            return request.risk_model_output
        if not request.athlete_metrics:
            return None
        try:
            if self.risk_predictor is None:
                self.risk_predictor = InjuryRiskPredictor()
            raw = self.risk_predictor.predict_one(request.athlete_metrics)
            return RiskModelOutput(
                injury_risk_probability=raw.get("injury_risk_probability"),
                fatigue_score=request.athlete_metrics.get("fatigue_index"),
                workload_spike=request.athlete_metrics.get("workload_spike"),
                sleep_quality=request.athlete_metrics.get("sleep_quality"),
                recovery_index=request.athlete_metrics.get("recovery_score"),
                raw=raw,
            )
        except Exception:
            return None

    def _rule_probability_proxy(self, match_score: float, profile: InjuryProfile, risk_output: Optional[RiskModelOutput]) -> float:
        base = 0.03 + 0.92 * match_score
        if risk_output and risk_output.injury_risk_probability is not None:
            risk_adjustment = (risk_output.injury_risk_probability - 0.5) * profile.risk_sensitivity
            base += risk_adjustment
        if risk_output and risk_output.fatigue_score is not None and risk_output.fatigue_score >= 75:
            base += min(profile.risk_sensitivity, 0.08)
        return min(max(base, 0.001), 0.999)

    def _combine_probabilities(self, request: DiagnosisRequest, scored: List[Tuple[str, InjuryProfile, float, float, str, List[str]]]) -> Dict[str, float]:
        rule_probs = {name: rule_score for name, _, _, rule_score, _, _ in scored}
        if self.ml_artifact is None:
            return _softmax(rule_probs)

        try:
            model = self.ml_artifact["model"]
            classes = list(self.ml_artifact["classes"])
            features = self._feature_vector(request, scored)
            ml_probs = dict(zip(classes, model.predict_proba([features])[0]))
            blended = {}
            for name in rule_probs:
                blended[name] = 0.55 * rule_probs[name] + 0.45 * ml_probs.get(name, 0.0)
            return _softmax(blended)
        except Exception:
            return _softmax(rule_probs)

    def _feature_vector(self, request: DiagnosisRequest, scored: List[Tuple[str, InjuryProfile, float, float, str, List[str]]]) -> List[float]:
        symptoms = request.symptoms
        mechanism = request.mechanism
        risk = request.risk_model_output
        base = [
            float(symptoms.pain_severity or 0),
            float(symptoms.swelling),
            float(symptoms.instability),
            float(symptoms.locking),
            float(symptoms.popping_sound),
            float(symptoms.reduced_range_of_motion),
            float(mechanism.twisting),
            float(mechanism.overuse),
            float(mechanism.direct_impact),
            float(mechanism.running),
            float(mechanism.throwing),
            float(mechanism.jumping),
            float(risk.injury_risk_probability if risk and risk.injury_risk_probability is not None else 0.0),
            float(risk.fatigue_score if risk and risk.fatigue_score is not None else 0.0),
        ]
        base.extend([match_score for _, _, match_score, _, _, _ in scored])
        return base

    def _confidence(self, predictions: List[InjuryPrediction]) -> float:
        if not predictions:
            return 0.0
        top = predictions[0].probability
        second = predictions[1].probability if len(predictions) > 1 else 0.0
        return min(max(top + 0.35 * max(top - second, 0.0), 0.0), 1.0)

    def _explanation(
        self,
        request: DiagnosisRequest,
        top_name: Optional[str],
        scored: List[Tuple[str, InjuryProfile, float, float, str, List[str]]],
        risk_output: Optional[RiskModelOutput],
    ) -> List[str]:
        if top_name is None:
            return []
        for name, profile, _, _, _, reasoning in scored:
            if name == top_name:
                output = ["Pain located in " + profile.region]
                output.extend(reasoning[:5])
                if risk_output and risk_output.injury_risk_probability is not None:
                    output.append("Risk model probability used as supporting context")
                if not output:
                    output = profile.explanation_terms[:4]
                return _dedupe(output)
        return []

    def _model_version(self) -> str:
        if self.ml_artifact:
            return str(self.ml_artifact.get("version", "diagnosis-ml"))
        return f"knowledge-{self.knowledge_base.version}"


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    names = list(scores)
    raw = np.array([scores[name] for name in names], dtype=float)
    raw = np.clip(raw, 1e-6, None)
    raw = raw * 5.0
    exp = np.exp(raw - raw.max())
    probs = exp / exp.sum()
    return dict(zip(names, probs))


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output
