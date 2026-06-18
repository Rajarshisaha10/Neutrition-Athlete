from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DriftSummary:
    feature: str
    reference_mean: float
    current_mean: float
    standardized_shift: float
    drift_flag: bool


def population_shift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    z_threshold: float = 3.0,
) -> list[DriftSummary]:
    """Simple numeric drift placeholder for production monitoring integrations."""
    numeric = [c for c in reference.select_dtypes(include=[np.number]).columns if c in current.columns]
    summaries: list[DriftSummary] = []
    for col in numeric:
        ref = pd.to_numeric(reference[col], errors="coerce")
        cur = pd.to_numeric(current[col], errors="coerce")
        ref_std = float(ref.std(skipna=True) or 0.0)
        shift = 0.0 if ref_std == 0.0 else float((cur.mean(skipna=True) - ref.mean(skipna=True)) / ref_std)
        summaries.append(
            DriftSummary(
                feature=col,
                reference_mean=float(ref.mean(skipna=True)),
                current_mean=float(cur.mean(skipna=True)),
                standardized_shift=shift,
                drift_flag=abs(shift) >= z_threshold,
            )
        )
    return summaries
