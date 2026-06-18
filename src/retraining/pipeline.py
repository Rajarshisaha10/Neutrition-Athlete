from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split

from src.database.repository import DiagnosisRepository
from src.utils.io import ensure_dir


class DiagnosisRetrainingPipeline:
    def __init__(self, repository: DiagnosisRepository | None = None, model_root: str | Path = "models/diagnosis"):
        self.repository = repository or DiagnosisRepository()
        self.model_root = Path(model_root)

    def build_labeled_dataset(self) -> pd.DataFrame:
        rows = []
        for case in self.repository.confirmed_cases():
            q = case["questionnaire_data"]
            symptoms = q.get("symptoms", {})
            mechanism = q.get("mechanism", {})
            risk = q.get("risk_model_output") or {}
            rows.append(
                {
                    "pain_severity": symptoms.get("pain_severity") or 0,
                    "swelling": int(bool(symptoms.get("swelling"))),
                    "instability": int(bool(symptoms.get("instability"))),
                    "locking": int(bool(symptoms.get("locking"))),
                    "popping_sound": int(bool(symptoms.get("popping_sound"))),
                    "reduced_range_of_motion": int(bool(symptoms.get("reduced_range_of_motion"))),
                    "twisting": int(bool(mechanism.get("twisting"))),
                    "overuse": int(bool(mechanism.get("overuse"))),
                    "direct_impact": int(bool(mechanism.get("direct_impact"))),
                    "running": int(bool(mechanism.get("running"))),
                    "throwing": int(bool(mechanism.get("throwing"))),
                    "jumping": int(bool(mechanism.get("jumping"))),
                    "injury_risk_probability": risk.get("injury_risk_probability") or 0,
                    "fatigue_score": risk.get("fatigue_score") or 0,
                    "confirmed_diagnosis": case["confirmed_diagnosis"],
                }
            )
        return pd.DataFrame(rows)

    def retrain(self, min_cases: int = 50) -> dict[str, Any]:
        data = self.build_labeled_dataset()
        if len(data) < min_cases:
            return {"status": "skipped", "reason": "not_enough_confirmed_cases", "case_count": int(len(data))}

        X = data.drop(columns=["confirmed_diagnosis"])
        y = data["confirmed_diagnosis"]
        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=42, stratify=stratify)
        holdout_stratify = y_holdout if y_holdout.value_counts().min() >= 2 else None
        X_cal, X_test, y_cal, y_test = train_test_split(X_holdout, y_holdout, test_size=0.5, random_state=42, stratify=holdout_stratify)

        candidates = self._candidate_models()
        scored = []
        for name, model in candidates.items():
            try:
                model.fit(X_train, y_train)
                for calibration in ("platt", "isotonic"):
                    calibrated = self._calibrate(model, X_cal, y_cal, calibration)
                    pred = calibrated.predict(X_test)
                    metrics = {
                        "model": name,
                        "calibration": calibration,
                        "accuracy": float(accuracy_score(y_test, pred)),
                        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
                        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                        "case_count": int(len(data)),
                    }
                    score = metrics["precision_macro"] + 0.35 * metrics["f1_macro"]
                    scored.append((score, calibrated, metrics))
            except Exception:
                continue
        if not scored:
            return {"status": "skipped", "reason": "all_candidate_models_failed", "case_count": int(len(data))}
        scored.sort(key=lambda item: item[0], reverse=True)
        model = scored[0][1]
        metrics = scored[0][2]
        version = "v" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
        version_dir = ensure_dir(self.model_root / version)
        artifact = {"model": model, "classes": list(model.classes_), "version": version, "metrics": metrics}
        joblib.dump(artifact, version_dir / "diagnosis_model.joblib")
        with (version_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        ensure_dir(self.model_root)
        joblib.dump(artifact, self.model_root / "latest.joblib")
        return {"status": "trained", "version": version, "metrics": metrics}

    def _candidate_models(self):
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                min_samples_leaf=2,
                random_state=42,
            )
        }
        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_estimators=250,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                random_state=42,
            )
        except Exception:
            pass
        try:
            from catboost import CatBoostClassifier

            models["catboost"] = CatBoostClassifier(
                loss_function="MultiClass",
                iterations=250,
                depth=4,
                learning_rate=0.05,
                auto_class_weights="Balanced",
                random_seed=42,
                verbose=False,
            )
        except Exception:
            pass
        return models

    def _calibrate(self, model, X_cal, y_cal, calibration: str):
        method = "sigmoid" if calibration == "platt" else "isotonic"
        try:
            calibrated = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
        except TypeError:  # pragma: no cover
            calibrated = CalibratedClassifierCV(base_estimator=model, method=method, cv="prefit")
        calibrated.fit(X_cal, y_cal)
        return calibrated
