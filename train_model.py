"""
AthleteEdge AI — XGBoost Model Trainer
=======================================
Trains one XGBoost regressor per macro target (calories, protein, carbs, fat)
and three binary classifiers for micronutrient flags (iron, calcium, vit-D).

Run:
    python train_model.py

Requires:
    data/nutrition_dataset.csv   ← run generate_dataset.py first

Output:
    models/xgb_calories.pkl
    models/xgb_protein.pkl
    models/xgb_carbs.pkl
    models/xgb_fat.pkl
    models/xgb_needs_iron.pkl
    models/xgb_needs_calcium.pkl
    models/xgb_needs_vitd.pkl
    models/feature_list.pkl      ← needed by meal_planner.py
"""

import os, joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

os.makedirs("models", exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/nutrition_dataset.csv")

FEATURES = [
    "sport_enc", "gender_enc", "age", "weight_kg", "height_cm",
    "training_intensity", "session_duration_min", "training_days_per_week",
    "is_match_day", "is_recovery_day", "fatigue_score",
    "goal_enc", "diet_enc",
]

REGRESSION_TARGETS   = ["calories_kcal", "protein_g", "carbs_g", "fat_g"]
CLASSIFICATION_TARGETS = ["needs_iron", "needs_calcium", "needs_vitd"]

X = df[FEATURES]
X_train, X_test = train_test_split(X, test_size=0.15, random_state=42)

print("=" * 60)
print("AthleteEdge AI — Model Training")
print("=" * 60)

# ── Regression models ────────────────────────────────────────────────────────
print("\n📊 Regression targets (MAE on test set):")
print("-" * 40)

for target in REGRESSION_TARGETS:
    y = df[target]
    y_train = y.iloc[X_train.index]
    y_test  = y.iloc[X_test.index]

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
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)

    fname = f"models/xgb_{target.replace('_kcal','').replace('_g','')}.pkl"
    joblib.dump(model, fname)
    print(f"  {target:<18}  MAE = {mae:.2f}   → saved {fname}")

# ── Classification models ────────────────────────────────────────────────────
print("\n🏷️  Micronutrient flags (accuracy on test set):")
print("-" * 40)

for target in CLASSIFICATION_TARGETS:
    y = df[target]
    y_train = y.iloc[X_train.index]
    y_test  = y.iloc[X_test.index]

    # Handle class imbalance
    pos   = y_train.sum()
    neg   = len(y_train) - pos
    scale = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    preds    = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    fname = f"models/xgb_{target}.pkl"
    joblib.dump(model, fname)
    print(f"  {target:<18}  Acc = {accuracy:.3f}  → saved {fname}")

# ── Save feature list ────────────────────────────────────────────────────────
joblib.dump(FEATURES, "models/feature_list.pkl")
print("\n✅  Feature list saved → models/feature_list.pkl")

# ── Quick sanity check ───────────────────────────────────────────────────────
print("\n🔍  Sanity check — Cricket fast bowler, 22 y/o, 72 kg, match day:")
sample = pd.DataFrame([{
    "sport_enc": 0, "gender_enc": 0, "age": 22, "weight_kg": 72,
    "height_cm": 175, "training_intensity": 8, "session_duration_min": 120,
    "training_days_per_week": 6, "is_match_day": 1, "is_recovery_day": 0,
    "fatigue_score": 6, "goal_enc": 0, "diet_enc": 1,
}])

cal_model  = joblib.load("models/xgb_calories.pkl")
prot_model = joblib.load("models/xgb_protein.pkl")
carb_model = joblib.load("models/xgb_carbs.pkl")
fat_model  = joblib.load("models/xgb_fat.pkl")

print(f"  Calories : {cal_model.predict(sample)[0]:.0f} kcal")
print(f"  Protein  : {prot_model.predict(sample)[0]:.1f} g")
print(f"  Carbs    : {carb_model.predict(sample)[0]:.1f} g")
print(f"  Fat      : {fat_model.predict(sample)[0]:.1f} g")

print("\n✅  All models trained and saved to models/")