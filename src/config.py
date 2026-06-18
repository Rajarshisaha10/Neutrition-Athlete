from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


try:
    import yaml
except Exception:  # pragma: no cover - optional until dependencies are installed
    yaml = None


@dataclass
class DataConfig:
    multimodal: str = "data/multimodal"
    sirp600: str = "data/sirp600"
    runners: str = "data/runners"
    optional: str = "data/optional"


@dataclass
class TrainingConfig:
    test_size: float = 0.15
    validation_size: float = 0.20
    tune_models: bool = True
    n_iter_search: int = 12
    cv_folds: int = 5
    use_gpu: bool = False
    gpu_device_id: int = 0
    gpu_only_models: bool = False
    include_logistic_regression: bool = False
    permutation_repeats: int = 2
    report_sample_size: int | None = 3000
    shap_sample_size: int = 100
    use_smote: bool = False
    optimize_with_optuna: bool = False
    enable_mlflow: bool = False
    sample_limit: int | None = None


@dataclass
class OutputConfig:
    models_dir: str = "models"
    reports_dir: str = "outputs"


@dataclass
class AppConfig:
    random_state: int = 42
    target_positive_rule: str = "any_positive"
    precision_minimum: float = 0.80
    max_false_positive_rate: float = 0.20
    threshold_grid_size: int = 199
    risk_thresholds: dict[str, float] = field(default_factory=lambda: {"low": 0.30, "high": 0.70})
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    @property
    def model_path(self) -> Path:
        root_dir = Path(__file__).resolve().parent.parent
        return root_dir / self.outputs.models_dir / "injury_risk_pipeline.joblib"


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        current = getattr(instance, key, None)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        elif hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path = None) -> AppConfig:
    config = AppConfig()
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    else:
        path = Path(path)
    if not path.exists():
        return config
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config.yaml. Install requirements.txt first.")
    with path.open("r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    return _merge_dataclass(config, values)
