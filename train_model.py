"""
AthleteEdge AI - XGBoost Model Trainer
======================================
Trains one XGBoost regressor per macro target and one XGBoost classifier per
micronutrient flag.

Run:
    python train_model.py

Requires:
    data/nutrition_dataset.csv

Outputs:
    models/xgb_calories.pkl
    models/xgb_protein.pkl
    models/xgb_carbs.pkl
    models/xgb_fat.pkl
    models/xgb_needs_iron.pkl
    models/xgb_needs_calcium.pkl
    models/xgb_needs_vitd.pkl
    models/feature_list.pkl
    output.txt
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


DATA_PATH = Path("data") / "nutrition_dataset.csv"
MODEL_DIR = Path("models")
REPORT_PATH = Path("output.txt")

FEATURES = [
    "sport_enc",
    "gender_enc",
    "age",
    "weight_kg",
    "height_cm",
    "training_intensity",
    "session_duration_min",
    "training_days_per_week",
    "is_match_day",
    "is_recovery_day",
    "fatigue_score",
    "goal_enc",
    "diet_enc",
]

REGRESSION_TARGETS = ["calories_kcal", "protein_g", "carbs_g", "fat_g"]
CLASSIFICATION_TARGETS = ["needs_iron", "needs_calcium", "needs_vitd"]


def log(line: str = "", report: list[str] | None = None) -> None:
    print(line)
    if report is not None:
        report.append(line)


def model_name_for_target(target: str) -> str:
    return f"xgb_{target.replace('_kcal', '').replace('_g', '')}.pkl"


def load_training_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def train_regressors(df: pd.DataFrame, x_train: pd.DataFrame, x_test: pd.DataFrame, report: list[str]) -> None:
    log("", report)
    log("Regression targets", report)
    log("-" * 70, report)
    log(f"{'target':<18} {'MAE':>10} {'MSE':>14} {'RMSE':>10}  saved_model", report)
    log("-" * 70, report)

    for target in REGRESSION_TARGETS:
        y = df[target]
        y_train = y.loc[x_train.index]
        y_test = y.loc[x_test.index]

        model = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(x_train, y_train)

        preds = model.predict(x_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5

        model_path = MODEL_DIR / model_name_for_target(target)
        joblib.dump(model, model_path)

        log(
            f"{target:<18} {mae:>10.2f} {mse:>14.2f} {rmse:>10.2f}  {model_path}",
            report,
        )


def train_classifiers(df: pd.DataFrame, x_train: pd.DataFrame, x_test: pd.DataFrame, report: list[str]) -> None:
    log("", report)
    log("Micronutrient flag classifiers", report)
    log("-" * 70, report)
    log(f"{'target':<18} {'accuracy':>10} {'f1_score':>10}  saved_model", report)
    log("-" * 70, report)

    for target in CLASSIFICATION_TARGETS:
        y = df[target]
        y_train = y.loc[x_train.index]
        y_test = y.loc[x_test.index]

        pos = y_train.sum()
        neg = len(y_train) - pos
        scale = neg / pos if pos > 0 else 1.0

        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(x_train, y_train)

        preds = model.predict(x_test)
        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)

        model_path = MODEL_DIR / f"xgb_{target}.pkl"
        joblib.dump(model, model_path)

        log(f"{target:<18} {accuracy:>10.3f} {f1:>10.3f}  {model_path}", report)


def add_sanity_check(report: list[str]) -> None:
    sample = pd.DataFrame(
        [
            {
                "sport_enc": 0,
                "gender_enc": 0,
                "age": 22,
                "weight_kg": 72,
                "height_cm": 175,
                "training_intensity": 8,
                "session_duration_min": 120,
                "training_days_per_week": 6,
                "is_match_day": 1,
                "is_recovery_day": 0,
                "fatigue_score": 6,
                "goal_enc": 0,
                "diet_enc": 1,
            }
        ]
    )

    cal_model = joblib.load(MODEL_DIR / "xgb_calories.pkl")
    prot_model = joblib.load(MODEL_DIR / "xgb_protein.pkl")
    carb_model = joblib.load(MODEL_DIR / "xgb_carbs.pkl")
    fat_model = joblib.load(MODEL_DIR / "xgb_fat.pkl")

    log("", report)
    log("Sanity check: Cricket fast bowler, 22 y/o, 72 kg, match day", report)
    log("-" * 70, report)
    log(f"Calories : {cal_model.predict(sample)[0]:.0f} kcal", report)
    log(f"Protein  : {prot_model.predict(sample)[0]:.1f} g", report)
    log(f"Carbs    : {carb_model.predict(sample)[0]:.1f} g", report)
    log(f"Fat      : {fat_model.predict(sample)[0]:.1f} g", report)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    report: list[str] = []
    df = load_training_data()
    x = df[FEATURES]
    x_train, x_test = train_test_split(x, test_size=0.15, random_state=42)

    log("=" * 70, report)
    log("AthleteEdge AI - Model Training Report", report)
    log("=" * 70, report)
    log(f"Dataset: {DATA_PATH}", report)
    log(f"Rows: {len(df)} | Features: {len(FEATURES)} | Test split: {len(x_test)} rows", report)

    train_regressors(df, x_train, x_test, report)
    train_classifiers(df, x_train, x_test, report)

    feature_path = MODEL_DIR / "feature_list.pkl"
    joblib.dump(FEATURES, feature_path)
    log("", report)
    log(f"Feature list saved: {feature_path}", report)

    add_sanity_check(report)

    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    log("", None)
    log(f"Training complete. Report saved to {REPORT_PATH}", None)


if __name__ == "__main__":
    main()
