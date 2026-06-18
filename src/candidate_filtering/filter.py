from __future__ import annotations

from typing import Dict, Tuple

from src.knowledge_engine.knowledge_base import InjuryKnowledgeBase, InjuryProfile
from src.questionnaire.schemas import DiagnosisRequest


REGION_ALIASES = {
    "hamstring": "hamstring",
    "thigh": "thigh",
    "shin": "shin",
    "calf": "calf",
    "ankle": "ankle",
    "foot": "ankle",
    "knee": "knee",
    "shoulder": "shoulder",
    "elbow": "elbow",
    "back": "back",
    "head": "head",
    "neck": "head",
}


def normalized_region(location: str) -> str:
    text = (location or "").strip().lower()
    return REGION_ALIASES.get(text, text)


class CandidateFilter:
    def __init__(self, knowledge_base: InjuryKnowledgeBase):
        self.knowledge_base = knowledge_base

    def filter(self, request: DiagnosisRequest) -> Dict[str, Tuple[InjuryProfile, float, str]]:
        requested_region = normalized_region(request.pain_location)
        output: Dict[str, Tuple[InjuryProfile, float, str]] = {}
        for profile in self.knowledge_base.all():
            multiplier = 1.0
            status = "candidate"

            if requested_region and profile.region != requested_region:
                compatible = requested_region == "thigh" and profile.region == "hamstring"
                if not compatible:
                    continue

            if request.onset and profile.onset and request.onset not in profile.onset:
                multiplier *= 0.35
                status = "reduced_onset_mismatch"

            if request.onset == "sudden" and "gradual" in profile.onset and "sudden" not in profile.onset:
                multiplier *= 0.25
                status = "reduced_overuse_pattern"

            if request.onset == "gradual" and "sudden" in profile.onset and "gradual" not in profile.onset:
                multiplier *= 0.45
                status = "reduced_acute_pattern"

            if "throwing" in profile.mechanisms and not request.mechanism.throwing and profile.region in {"shoulder", "elbow"}:
                multiplier *= 0.70
                status = "reduced_throwing_absent"

            output[profile.name] = (profile, multiplier, status)
        return output
