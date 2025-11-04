"""Compatibility wrappers around the core PCAP modeling helpers."""

from __future__ import annotations

import json
from datetime import datetime
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
from .vectorizer import numeric_feature_names

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

    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    stamp_token = timestamp.strftime("%Y%m%d_%H%M%S")

    numeric_names = numeric_feature_names()

    metadata: Dict[str, object] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "timestamp": timestamp_str,
        "pipeline_latest": model_path.name,
        "pipeline_path": model_path.name,
        "feature_order": list(numeric_names),
        "feature_names_in": list(numeric_names),
        "feature_columns": summary.feature_names,
        "label_mapping": summary.label_mapping or {},
        "contamination": kwargs.get("contamination"),
        "training_anomaly_ratio": kwargs.get("training_anomaly_ratio"),
    }

    metadata_path = models_root / f"iforest_metadata_{stamp_token}.json"
    latest_metadata_path = models_root / "latest_iforest_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    with latest_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    result: Dict[str, object] = {
        "model_path": str(model_path),
        "pipeline_path": str(model_path),
        "pipeline_latest": str(model_path),
        "metadata_path": str(metadata_path),
        "metadata_latest": str(latest_metadata_path),
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
        "timestamp": timestamp_str,
        "schema_version": MODEL_SCHEMA_VERSION,
        "summary": summary,
        "metadata": metadata,
    }

    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    return result
