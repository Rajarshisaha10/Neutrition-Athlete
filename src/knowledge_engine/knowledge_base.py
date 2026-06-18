from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class InjuryProfile:
    name: str
    region: str
    onset: List[str]
    mechanisms: Dict[str, float]
    symptoms: Dict[str, float]
    risk_sensitivity: float
    explanation_terms: List[str]
    red_flags: List[str]


class InjuryKnowledgeBase:
    def __init__(self, path: str | Path = None):
        if path is None:
            # Go up two directories from src/knowledge_engine to root, then knowledge_base
            root_dir = Path(__file__).resolve().parent.parent.parent
            path = root_dir / "knowledge_base" / "injuries_v1.json"
        self.path = Path(path)
        self.version = "v1"
        self.injuries = self._load()

    def _load(self) -> Dict[str, InjuryProfile]:
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.version = data.get("version", "v1")
        profiles: Dict[str, InjuryProfile] = {}
        for name, raw in data["injuries"].items():
            profiles[name] = InjuryProfile(
                name=name,
                region=raw["region"],
                onset=list(raw.get("onset", [])),
                mechanisms=dict(raw.get("mechanisms", {})),
                symptoms=dict(raw.get("symptoms", {})),
                risk_sensitivity=float(raw.get("risk_sensitivity", 0.0)),
                explanation_terms=list(raw.get("explanation_terms", [])),
                red_flags=list(raw.get("red_flags", [])),
            )
        return profiles

    def all(self) -> List[InjuryProfile]:
        return list(self.injuries.values())

    def get(self, name: str) -> InjuryProfile:
        return self.injuries[name]

    def supported_injuries(self) -> List[str]:
        return sorted(self.injuries)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "injuries": {
                name: {
                    "region": profile.region,
                    "onset": profile.onset,
                    "mechanisms": profile.mechanisms,
                    "symptoms": profile.symptoms,
                }
                for name, profile in self.injuries.items()
            },
        }
