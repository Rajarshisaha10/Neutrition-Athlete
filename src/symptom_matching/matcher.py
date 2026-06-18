from __future__ import annotations

from typing import Dict, List, Tuple

from src.knowledge_engine.knowledge_base import InjuryProfile
from src.questionnaire.schemas import DiagnosisRequest


class SymptomMatcher:
    def score(self, request: DiagnosisRequest, profile: InjuryProfile) -> Tuple[float, List[str]]:
        matched = 0.0
        possible = sum(profile.symptoms.values()) + sum(profile.mechanisms.values())
        reasoning: List[str] = []

        symptoms = request.symptoms
        mechanism = request.mechanism

        for name, weight in profile.symptoms.items():
            value = getattr(symptoms, name, False)
            if name == "weight_bearing_difficulty":
                value = (symptoms.weight_bearing_ability or "").lower() in {"difficult", "unable", "limited"}
            elif name == "pain_severity":
                value = symptoms.pain_severity is not None and symptoms.pain_severity >= 6
            if value:
                matched += weight
                reasoning.append(_humanize(name))

        for name, weight in profile.mechanisms.items():
            if getattr(mechanism, name, False):
                matched += weight
                reasoning.append(_humanize(name) + " mechanism")

        if request.onset and request.onset in profile.onset:
            matched += 1.0
            possible += 1.0
            reasoning.append(_humanize(request.onset) + " onset")

        if profile.region == request.pain_location.lower():
            matched += 1.5
            possible += 1.5
            reasoning.append("Pain located in " + profile.region)

        normalized = matched / max(possible, 1.0)
        return min(max(normalized, 0.0), 1.0), _dedupe(reasoning)


def _humanize(value: str) -> str:
    return value.replace("_", " ").capitalize()


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def build_match_features(request: DiagnosisRequest, candidates: Dict[str, InjuryProfile]) -> Dict[str, float]:
    matcher = SymptomMatcher()
    return {name: matcher.score(request, profile)[0] for name, profile in candidates.items()}
