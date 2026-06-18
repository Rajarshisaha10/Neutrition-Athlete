from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)

TARGET_CANDIDATES = [
    "injury_risk",
    "injury_occurred",
    "Likelihood_of_Injury",
    "likelihood_of_injury",
    "injury",
    "injured",
    "target",
]

ID_CANDIDATES = ["athlete_key", "athlete_id", "Athlete ID", "player_id", "session_id", "id"]
TIME_CANDIDATES = ["event_time", "Date", "date", "timestamp", "session_date", "time", "session_id"]


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset file: {path}")


def _dataset_files(root: str) -> list[Path]:
    path = Path(root)
    if not path.exists():
        return []
    files = sorted(list(path.glob("*.csv")) + list(path.glob("*.xlsx")) + list(path.glob("*.xls")))
    return [p for p in files if not p.name.startswith("~$")]


def find_target_column(df: pd.DataFrame) -> str:
    normalized = {c.lower(): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    injury_like = [c for c in df.columns if "injury" in c.lower()]
    if injury_like:
        return injury_like[-1]
    raise ValueError(f"No target column found. Columns: {list(df.columns)}")


def normalize_target(series: pd.Series) -> pd.Series:
    if series.dtype == "O":
        lowered = series.astype(str).str.strip().str.lower()
        positive = lowered.isin({"1", "true", "yes", "y", "risk", "injured", "high", "moderate", "medium", "low", "low risk", "injury"})
        numeric = pd.to_numeric(series, errors="coerce")
        return ((numeric.fillna(0) > 0) | positive).astype(int)
    return (pd.to_numeric(series, errors="coerce").fillna(0) > 0).astype(int)


def load_datasets(config: AppConfig) -> pd.DataFrame:
    roots = {
        "multimodal": config.data.multimodal,
        "sirp600": config.data.sirp600,
        "runners": config.data.runners,
        "optional": config.data.optional,
    }
    frames: list[pd.DataFrame] = []
    for source, root in roots.items():
        for file_path in _dataset_files(root):
            LOGGER.info("Loading %s", file_path)
            df = _read_file(file_path)
            if df.empty:
                continue
            target = find_target_column(df)
            df = df.copy()
            df["injury_risk_target"] = normalize_target(df[target])
            if target != "injury_risk_target":
                df = df.drop(columns=[target])
            df["source_dataset"] = source
            df["source_file"] = file_path.name
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No supported datasets found under configured data directories.")
    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    id_sources = [c for c in ["athlete_id", "Athlete ID", "player_id", "id"] if c in combined.columns]
    if id_sources:
        raw_id = combined[id_sources].bfill(axis=1).iloc[:, 0].astype(str)
        combined["athlete_key"] = combined["source_dataset"].astype(str) + ":" + raw_id
    time_sources = [c for c in ["Date", "date", "timestamp", "session_date", "time", "session_id"] if c in combined.columns]
    if time_sources:
        combined["event_time"] = combined[time_sources].bfill(axis=1).iloc[:, 0]
    LOGGER.info("Combined dataset shape: %s", combined.shape)
    return combined


def infer_temporal_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    id_col = next((c for c in ID_CANDIDATES if c in df.columns), None)
    time_col = next((c for c in TIME_CANDIDATES if c in df.columns), None)
    return id_col, time_col
