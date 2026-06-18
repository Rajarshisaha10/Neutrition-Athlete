from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.io import ensure_dir


RISK_FACTOR_ALIASES = {
    "fatigue": "elevated fatigue trend",
    "sleep": "sleep reduction",
    "workload": "high workload",
    "training_load": "high workload",
    "training load": "high workload",
    "recovery": "reduced recovery",
    "stress": "elevated stress",
    "exertion": "high exertion",
    "heart_rate": "elevated heart rate",
    "hydration": "reduced hydration",
}

EXPLANATION_EXCLUDE = ("date", "time", "source", "file", "id", "key", "session")


def humanize_factor(feature: str) -> str:
    clean = feature.replace("_", " ").replace(".", " ")
    lowered = clean.lower()
    for key, label in RISK_FACTOR_ALIASES.items():
        if key in lowered:
            return label
    return clean


def permutation_importance_report(
    model,
    X,
    y,
    feature_names: list[str],
    output_dir: str | Path,
    n_repeats: int = 2,
) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, X, y, scoring="average_precision", n_repeats=n_repeats, random_state=42, n_jobs=1)
    df = pd.DataFrame({"feature": feature_names, "importance": result.importances_mean}).sort_values("importance", ascending=False)
    output_dir = ensure_dir(output_dir)
    df.to_csv(output_dir / "feature_importance.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        top = df.head(25).iloc[::-1]
        plt.figure(figsize=(8, 7))
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("Permutation Importance")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=160)
        plt.close()
    except Exception as exc:
        with (output_dir / "feature_importance_plot_unavailable.txt").open("w", encoding="utf-8") as f:
            f.write(f"Feature importance plot could not be generated: {exc}\n")
    return df


def shap_report(model, X_sample, feature_names: list[str], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    try:
        import matplotlib.pyplot as plt
        import shap

        explainer = shap.Explainer(model.predict_proba, X_sample, feature_names=feature_names)
        values = explainer(X_sample)
        shap_values = values.values[:, :, 1] if values.values.ndim == 3 else values.values
        shap.summary_plot(shap_values, features=X_sample, feature_names=feature_names, show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_global_importance.png", dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        with (output_dir / "shap_unavailable.txt").open("w", encoding="utf-8") as f:
            f.write(f"SHAP report could not be generated: {exc}\n")


def top_local_factors(model, X_row, feature_names: list[str], top_n: int = 3) -> list[str]:
    try:
        baseline = np.zeros_like(X_row, dtype=float)
        base_prob = model.predict_proba(baseline)[:, 1][0]
        impacts: list[tuple[str, float]] = []
        for idx, name in enumerate(feature_names):
            probe = baseline.copy()
            probe[:, idx] = X_row[:, idx]
            impact = model.predict_proba(probe)[:, 1][0] - base_prob
            impacts.append((name, impact))
        return [humanize_factor(name) for name, impact in sorted(impacts, key=lambda x: x[1], reverse=True)[:top_n]]
    except Exception:
        return []


def top_raw_factors(model, row: pd.DataFrame, reference: pd.DataFrame, feature_names: list[str], top_n: int = 3) -> list[str]:
    """Estimate local factor impact by replacing one raw feature at a time with a reference value."""
    try:
        base = row.copy()
        baseline_prob = model.predict_proba(base)[:, 1][0]
        impacts: list[tuple[str, float]] = []
        reference_values = reference.median(numeric_only=True).to_dict()
        for col in feature_names:
            if any(token in col.lower() for token in EXPLANATION_EXCLUDE):
                continue
            if col not in base.columns:
                continue
            probe = base.copy()
            if col in reference_values:
                probe[col] = reference_values[col]
            else:
                mode = reference[col].mode(dropna=True) if col in reference.columns else pd.Series(dtype=object)
                probe[col] = mode.iloc[0] if len(mode) else pd.NA
            impact = baseline_prob - model.predict_proba(probe)[:, 1][0]
            impacts.append((col, impact))
        positive = [(name, impact) for name, impact in impacts if impact > 0]
        ranked = positive or impacts
        return [humanize_factor(name) for name, _ in sorted(ranked, key=lambda x: x[1], reverse=True)[:top_n]]
    except Exception:
        return []
