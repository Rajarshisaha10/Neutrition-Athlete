from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.io import ensure_dir


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def classification_metrics(y_true, y_prob, threshold: float) -> dict[str, float]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / max(fp + tn, 1)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "false_positive_rate": float(fpr),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def optimize_threshold(y_true, y_prob, min_precision: float, max_fpr: float, grid_size: int = 199) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_score = -np.inf
    best_metrics: dict[str, float] = {}
    for threshold in np.linspace(0.01, 0.99, grid_size):
        metrics = classification_metrics(y_true, y_prob, threshold)
        precision_penalty = max(0.0, min_precision - metrics["precision"]) * 2.0
        fpr_penalty = max(0.0, metrics["false_positive_rate"] - max_fpr) * 2.0
        score = metrics["precision"] + 0.35 * metrics["f1"] + 0.15 * metrics["pr_auc"] - precision_penalty - fpr_penalty
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
            best_metrics = metrics
    return best_threshold, best_metrics


def save_evaluation_plots(y_true, y_prob, threshold: float, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        with (output_dir / "plots_unavailable.txt").open("w", encoding="utf-8") as f:
            f.write(f"Evaluation plots could not be generated: {exc}\n")
        return
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Risk", "Risk"], yticklabels=["No Risk", "Risk"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close()

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=160)
    plt.close()


def write_metrics_report(metrics: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    with (output_dir / "metrics_report.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def save_model_comparison(rows: list[dict[str, Any]], output_dir: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values(["selection_score"], ascending=False)
    df.to_csv(ensure_dir(output_dir) / "model_comparison.csv", index=False)
    return df
