from __future__ import annotations

import re

import numpy as np
import pandas as pd


LEAKAGE_PATTERNS = re.compile(r"(injury|target|label|outcome)", re.IGNORECASE)
TREND_PATTERNS = re.compile(r"(fatigue|workload|load|sleep|recovery|exertion|stress|training)", re.IGNORECASE)


def _safe_numeric_columns(df: pd.DataFrame, id_col: str | None, time_col: str | None) -> list[str]:
    excluded = {"injury_risk_target", "source_dataset", "source_file"}
    if id_col:
        excluded.add(id_col)
    if time_col:
        excluded.add(time_col)
    return [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in excluded and not LEAKAGE_PATTERNS.search(c)
    ]


def add_temporal_features(
    df: pd.DataFrame,
    id_col: str | None,
    time_col: str | None,
    windows: tuple[int, ...] = (3, 7),
) -> pd.DataFrame:
    """Create shifted rolling features within athletes to prevent current-row leakage."""
    if id_col is None or id_col not in df.columns:
        return df.copy()

    out = df.copy()
    if time_col and time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        out = out.sort_values([id_col, time_col], kind="mergesort")
    else:
        out = out.sort_values([id_col], kind="mergesort")

    numeric_cols = _safe_numeric_columns(out, id_col, time_col)
    selected = [c for c in numeric_cols if TREND_PATTERNS.search(c)]
    selected = selected[:24] if selected else numeric_cols[:16]

    grouped = out.groupby(id_col, group_keys=False)
    derived: dict[str, pd.Series] = {}
    for col in selected:
        shifted = grouped[col].shift(1)
        derived[f"{col}_lag_1"] = shifted
        for window in windows:
            derived[f"{col}_roll{window}_mean"] = shifted.groupby(out[id_col]).rolling(window, min_periods=2).mean().reset_index(level=0, drop=True)
            derived[f"{col}_roll{window}_std"] = shifted.groupby(out[id_col]).rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
        derived[f"{col}_change_rate"] = grouped[col].pct_change().replace([np.inf, -np.inf], np.nan)

    if derived:
        out = pd.concat([out, pd.DataFrame(derived, index=out.index)], axis=1)

    if {"training_load", "recovery_score"}.issubset(out.columns):
        out["load_recovery_ratio"] = out["training_load"] / (out["recovery_score"].abs() + 1.0)
    if {"sleep_quality", "fatigue_index"}.issubset(out.columns):
        out["fatigue_sleep_pressure"] = out["fatigue_index"] / (out["sleep_quality"].abs() + 1.0)
    return out


def add_static_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pairs = [
        ("training_intensity", "training_duration", "intensity_duration_load"),
        ("Training_Intensity", "Recovery_Time", "intensity_recovery_pressure"),
        ("stress_level", "sleep_quality", "stress_sleep_pressure"),
        ("heart_rate", "recovery_score", "heart_recovery_pressure"),
    ]
    for left, right, name in pairs:
        if left in out.columns and right in out.columns:
            out[name] = pd.to_numeric(out[left], errors="coerce") / (pd.to_numeric(out[right], errors="coerce").abs() + 1.0)
    return out
