"""Compatibility wrappers around the core PCAP modeling helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from .modeling import (
    DEFAULT_MODEL_PARAMS,
    DetectionResult,
    TrainingSummary,
    compute_risk_components,
    detect_pcap_with_model,
    summarize_prediction_labels,
    train_hist_gradient_boosting,
)

__all__ = [
    "DEFAULT_MODEL_PARAMS",
    "TrainingSummary",
    "DetectionResult",
    "MODEL_SCHEMA_VERSION",
    "META_COLUMNS",
    "train_hist_gradient_boosting",
    "train_unsupervised_on_split",
    "detect_pcap_with_model",
    "summarize_prediction_labels",
    "compute_risk_components",
]

MODEL_SCHEMA_VERSION = "2025.10"

# Columns that should be preserved when presenting prediction results in the UI.
META_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Protocol",
    "Timestamp",
}


def _resolve_dataset_path(input_path: Union[str, Path]) -> Path:
    """Resolve the dataset CSV path from a user-provided input."""

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到训练数据: {input_path}")

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError("目前仅支持以 CSV 形式提供的训练数据。")
        return path

    candidates: List[Path] = []
    preferred_tokens = ("vectorized", "feature", "train")
    for child in path.glob("*.csv"):
        lowered = child.name.lower()
        if any(token in lowered for token in preferred_tokens):
            candidates.append(child)
    if not candidates:
        candidates = list(path.glob("*.csv"))

    if not candidates:
        raise FileNotFoundError("指定目录中未找到任何 CSV 训练数据。")

    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def train_unsupervised_on_split(
    input_path: Union[str, Path],
    results_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    **kwargs: Union[int, float, None],
) -> Dict[str, object]:
    """Wrapper used by the CLI/UI training entry points."""

    dataset_path = _resolve_dataset_path(input_path)
    models_root = Path(models_dir) if models_dir else dataset_path.parent
    models_root.mkdir(parents=True, exist_ok=True)

    model_path = models_root / "model.joblib"

    model_kwargs: Dict[str, Union[int, float, None]] = {}
    for key in DEFAULT_MODEL_PARAMS:
        if key in kwargs and kwargs[key] is not None:
            model_kwargs[key] = kwargs[key]

    summary = train_hist_gradient_boosting(dataset_path, model_path, **model_kwargs)

    result: Dict[str, object] = {
        "model_path": str(model_path),
        "pipeline_path": str(model_path),
        "pipeline_latest": str(model_path),
        "metadata_path": None,
        "metadata_latest": None,
        "model_joblib": str(model_path),
        "results_csv": None,
        "summary_csv": None,
        "scaler_path": None,
        "flows": summary.flow_count,
        "malicious": 0,
        "feature_columns": summary.feature_names,
        "classes": summary.classes,
        "label_mapping": summary.label_mapping,
        "dropped_flows": summary.dropped_flows,
        "timestamp": None,
        "schema_version": MODEL_SCHEMA_VERSION,
        "summary": summary,
        "metadata": None,
    }

    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    return result