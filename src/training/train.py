from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.config import AppConfig, load_config
from src.evaluation.explainability import permutation_importance_report, shap_report
from src.evaluation.metrics import (
    classification_metrics,
    optimize_threshold,
    save_evaluation_plots,
    save_model_comparison,
    write_metrics_report,
)
from src.feature_engineering.time_series import add_static_interactions, add_temporal_features
from src.preprocessing.data_ingestion import infer_temporal_columns, load_datasets
from src.preprocessing.pipeline import build_preprocessor
from src.training.ensemble import WeightedEnsembleClassifier
from src.utils.io import ensure_dir
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def chronological_split(df: pd.DataFrame, config: AppConfig, time_col: str | None):
    if time_col and time_col in df.columns and df[time_col].notna().sum() > 0:
        temporal_mask = df[time_col].notna()
        temporal = df.loc[temporal_mask].sort_values(time_col, kind="mergesort")
        non_temporal = df.loc[~temporal_mask]

        temporal_train = temporal_val = temporal_test = temporal.iloc[0:0]
        if len(temporal):
            n = len(temporal)
            test_start = int(n * (1 - config.training.test_size))
            val_start = int(test_start * (1 - config.training.validation_size))
            temporal_train = temporal.iloc[:val_start]
            temporal_val = temporal.iloc[val_start:test_start]
            temporal_test = temporal.iloc[test_start:]

        if len(non_temporal):
            non_train, non_val, non_test = stratified_split(non_temporal, config)
            return (
                pd.concat([temporal_train, non_train], axis=0),
                pd.concat([temporal_val, non_val], axis=0),
                pd.concat([temporal_test, non_test], axis=0),
            )
        return temporal_train, temporal_val, temporal_test

    return stratified_split(df, config)


def stratified_split(df: pd.DataFrame, config: AppConfig):
    train_val, test = train_test_split(
        df,
        test_size=config.training.test_size,
        stratify=df["injury_risk_target"],
        random_state=config.random_state,
    )
    train, val = train_test_split(
        train_val,
        test_size=config.training.validation_size,
        stratify=train_val["injury_risk_target"],
        random_state=config.random_state,
    )
    return train, val, test


def candidate_models(
    random_state: int,
    pos_weight: float,
    use_gpu: bool = False,
    gpu_device_id: int = 0,
    gpu_only_models: bool = False,
    include_logistic_regression: bool = False,
) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    models: dict[str, tuple[Any, dict[str, list[Any]]]] = {}
    try:
        from xgboost import XGBClassifier

        xgb_kwargs: dict[str, Any] = {}
        if use_gpu:
            xgb_kwargs = {"tree_method": "hist", "device": f"cuda:{gpu_device_id}"}
        models["xgboost"] = (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=pos_weight,
                random_state=random_state,
                n_jobs=1,
                **xgb_kwargs,
            ),
            {
                "classifier__n_estimators": [150, 250, 400],
                "classifier__max_depth": [2, 3, 5],
                "classifier__learning_rate": [0.02, 0.05, 0.1],
                "classifier__subsample": [0.7, 0.9],
            },
        )
    except Exception as exc:
        LOGGER.info("XGBoost unavailable; skipping. Import error: %s", exc)
    try:
        from lightgbm import LGBMClassifier

        lgbm_kwargs: dict[str, Any] = {}
        if use_gpu:
            lgbm_kwargs = {"device_type": "gpu", "gpu_device_id": gpu_device_id}
        models["lightgbm"] = (
            LGBMClassifier(
                objective="binary",
                class_weight={0: 1.0, 1: pos_weight},
                random_state=random_state,
                n_jobs=1,
                verbosity=-1,
                **lgbm_kwargs,
            ),
            {
                "classifier__n_estimators": [150, 250, 400],
                "classifier__num_leaves": [15, 31, 63],
                "classifier__learning_rate": [0.02, 0.05, 0.1],
                "classifier__subsample": [0.7, 0.9],
            },
        )
    except Exception as exc:
        LOGGER.info("LightGBM unavailable; skipping. Import error: %s", exc)
    try:
        from catboost import CatBoostClassifier

        cat_kwargs: dict[str, Any] = {}
        if use_gpu:
            cat_kwargs = {"task_type": "GPU", "devices": str(gpu_device_id)}
        models["catboost"] = (
            CatBoostClassifier(
                loss_function="Logloss",
                auto_class_weights="Balanced",
                random_seed=random_state,
                verbose=False,
                **cat_kwargs,
            ),
            {
                "classifier__iterations": [150, 250, 400],
                "classifier__depth": [3, 5, 7],
                "classifier__learning_rate": [0.02, 0.05, 0.1],
            },
        )
    except Exception as exc:
        LOGGER.info("CatBoost unavailable; skipping. Import error: %s", exc)
    if (not gpu_only_models) or include_logistic_regression:
        models["logistic_regression"] = (
            LogisticRegression(max_iter=3000, class_weight={0: 1.0, 1: pos_weight}, solver="liblinear", random_state=random_state),
            {"classifier__C": [0.05, 0.1, 0.5, 1.0, 2.0]},
        )
    if not gpu_only_models:
        models["random_forest"] = (
            RandomForestClassifier(
                n_estimators=350,
                min_samples_leaf=3,
                class_weight={0: 1.0, 1: pos_weight},
                random_state=random_state,
                n_jobs=1,
            ),
            {
                "classifier__max_depth": [4, 7, 10, None],
                "classifier__min_samples_leaf": [2, 4, 8],
                "classifier__max_features": ["sqrt", 0.5, None],
            },
        )
    return models


def gpu_accelerated_model_names() -> set[str]:
    return {"xgboost", "lightgbm", "catboost"}


def _calibrate(estimator, X_val, y_val, method: str):
    try:
        calibrated = CalibratedClassifierCV(estimator=estimator, method=method, cv="prefit")
    except TypeError:  # pragma: no cover - older sklearn
        calibrated = CalibratedClassifierCV(base_estimator=estimator, method=method, cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


def train_candidate(name: str, estimator, params: dict[str, list[Any]], X_train, y_train, config: AppConfig):
    pipe = Pipeline([("preprocessor", build_preprocessor(X_train)), ("classifier", estimator)])
    if config.training.tune_models and params:
        cv = StratifiedKFold(n_splits=config.training.cv_folds, shuffle=True, random_state=config.random_state)
        search = RandomizedSearchCV(
            pipe,
            params,
            n_iter=min(config.training.n_iter_search, max(1, np.prod([len(v) for v in params.values()]))),
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            random_state=config.random_state,
            refit=True,
        )
        search.fit(X_train, y_train)
        LOGGER.info("%s best params: %s", name, search.best_params_)
        return search.best_estimator_
    pipe.fit(X_train, y_train)
    return pipe


def train(config: AppConfig) -> dict[str, Any]:
    ensure_dir(config.outputs.models_dir)
    ensure_dir(config.outputs.reports_dir)

    raw = load_datasets(config)
    if config.training.sample_limit:
        raw = raw.groupby("injury_risk_target", group_keys=False).apply(
            lambda part: part.sample(
                min(len(part), max(1, int(config.training.sample_limit * len(part) / len(raw)))),
                random_state=config.random_state,
            )
        )
        raw = raw.sample(frac=1.0, random_state=config.random_state).reset_index(drop=True)
        LOGGER.info("Using sample-limited training frame: %s", raw.shape)
    id_col, time_col = infer_temporal_columns(raw)
    engineered = add_static_interactions(add_temporal_features(raw, id_col, time_col))
    drop_cols = ["injury_risk_target"]
    y = engineered["injury_risk_target"].astype(int)
    X = engineered.drop(columns=drop_cols)

    # Keep IDs/time for temporal sorting but remove direct identifiers from the model matrix.
    split_df = X.copy()
    split_df["injury_risk_target"] = y
    train_df, val_df, test_df = chronological_split(split_df, config, time_col)

    remove_from_features = {"source_file"}
    if id_col:
        remove_from_features.add(id_col)
    if time_col:
        remove_from_features.add(time_col)
    feature_cols = [c for c in X.columns if c not in remove_from_features]

    X_train, y_train = train_df[feature_cols], train_df["injury_risk_target"].astype(int)
    X_val, y_val = val_df[feature_cols], val_df["injury_risk_target"].astype(int)
    X_test, y_test = test_df[feature_cols], test_df["injury_risk_target"].astype(int)

    pos = max(int(y_train.sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    pos_weight = min(max(neg / pos, 1.0), 25.0)
    rows: list[dict[str, Any]] = []
    calibrated_models: list[tuple[str, Any, float]] = []
    candidates = candidate_models(
        config.random_state,
        pos_weight,
        use_gpu=config.training.use_gpu,
        gpu_device_id=config.training.gpu_device_id,
        gpu_only_models=config.training.gpu_only_models,
        include_logistic_regression=config.training.include_logistic_regression,
    )
    if not candidates:
        raise RuntimeError("No models are available. Install GPU booster libraries or set training.gpu_only_models: false.")
    if config.training.use_gpu:
        active = sorted(set(candidates) & gpu_accelerated_model_names())
        LOGGER.info("GPU training requested for available booster models: %s", active)
    if config.training.gpu_only_models:
        if config.training.include_logistic_regression:
            LOGGER.info("GPU-only model mode enabled with Logistic Regression baseline; skipping Random Forest.")
        else:
            LOGGER.info("GPU-only model mode enabled; skipping Logistic Regression and Random Forest.")

    for name, (estimator, params) in candidates.items():
        LOGGER.info("Training %s", name)
        try:
            fitted = train_candidate(name, estimator, params, X_train, y_train, config)
        except Exception as exc:
            if config.training.use_gpu and name in gpu_accelerated_model_names():
                LOGGER.warning("%s GPU training failed: %s", name, exc)
                LOGGER.warning("Retrying %s on CPU so the full run can finish.", name)
                cpu_estimator, cpu_params = candidate_models(
                    config.random_state,
                    pos_weight,
                    use_gpu=False,
                    gpu_only_models=False,
                    include_logistic_regression=True,
                )[name]
                fitted = train_candidate(name, cpu_estimator, cpu_params, X_train, y_train, config)
            else:
                raise
        calibration_options = {
            "uncalibrated": fitted,
            "platt": _calibrate(fitted, X_val, y_val, "sigmoid"),
            "isotonic": _calibrate(fitted, X_val, y_val, "isotonic"),
        }
        for cal_name, model in calibration_options.items():
            prob = model.predict_proba(X_val)[:, 1]
            threshold, metrics = optimize_threshold(
                y_val,
                prob,
                min_precision=config.precision_minimum,
                max_fpr=config.max_false_positive_rate,
                grid_size=config.threshold_grid_size,
            )
            selection_score = metrics["precision"] + 0.25 * metrics["f1"] + 0.15 * metrics["pr_auc"] - 0.50 * metrics["false_positive_rate"] - 0.25 * metrics["ece"]
            rows.append({"model": name, "calibration": cal_name, "selection_score": selection_score, **metrics})
            calibrated_models.append((f"{name}_{cal_name}", model, max(selection_score, 0.001)))

    comparison = save_model_comparison(rows, config.outputs.reports_dir)
    top_names = set(comparison.head(3).apply(lambda r: f"{r['model']}_{r['calibration']}", axis=1).tolist())
    ensemble_members = [item for item in calibrated_models if item[0] in top_names]
    if len(ensemble_members) > 1:
        ensemble = WeightedEnsembleClassifier(ensemble_members)
        val_prob = ensemble.predict_proba(X_val)[:, 1]
        threshold, val_metrics = optimize_threshold(
            y_val,
            val_prob,
            config.precision_minimum,
            config.max_false_positive_rate,
            config.threshold_grid_size,
        )
        final_model = ensemble
        final_name = "weighted_ensemble"
    else:
        best_row = comparison.iloc[0]
        final_name = f"{best_row['model']}_{best_row['calibration']}"
        final_model = next(model for name, model, _ in calibrated_models if name == final_name)
        val_prob = final_model.predict_proba(X_val)[:, 1]
        threshold, val_metrics = optimize_threshold(y_val, val_prob, config.precision_minimum, config.max_false_positive_rate, config.threshold_grid_size)

    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_prob, threshold)
    save_evaluation_plots(y_test, test_prob, threshold, config.outputs.reports_dir)

    X_report = X_test
    y_report = y_test
    if config.training.report_sample_size and len(X_report) > config.training.report_sample_size:
        report_idx = X_report.sample(config.training.report_sample_size, random_state=config.random_state).index
        X_report = X_report.loc[report_idx]
        y_report = y_report.loc[report_idx]
        LOGGER.info("Using %s held-out rows for permutation importance reports.", len(X_report))
    permutation_importance_report(
        final_model,
        X_report,
        y_report,
        feature_cols,
        config.outputs.reports_dir,
        n_repeats=config.training.permutation_repeats,
    )
    shap_rows = min(len(X_train), config.training.shap_sample_size)
    shap_report(final_model, X_train.head(shap_rows), feature_cols, config.outputs.reports_dir)

    artifact = {
        "model": final_model,
        "threshold": threshold,
        "risk_thresholds": config.risk_thresholds,
        "feature_columns": feature_cols,
        "raw_feature_columns": list(raw.columns),
        "id_col": id_col,
        "time_col": time_col,
        "selected_model": final_name,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "reference_frame": X_train.sample(min(len(X_train), 500), random_state=config.random_state),
    }
    joblib.dump(artifact, config.model_path)
    write_metrics_report(
        {
            "selected_model": final_name,
            "decision_threshold": threshold,
            "validation": val_metrics,
            "test": test_metrics,
            "positive_rate_train": float(y_train.mean()),
        },
        config.outputs.reports_dir,
    )
    LOGGER.info("Saved artifact to %s", config.model_path)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
