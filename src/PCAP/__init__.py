"""PCAP modeling package aggregating training and inference helpers."""

from .modeling import (
    DEFAULT_MODEL_PARAMS,
    DetectionResult,
    MODEL_SCHEMA_VERSION,
    META_COLUMNS,
    TrainingSummary,
    compute_risk_components,
    detect_pcap_with_model,
    summarize_prediction_labels,
    train_hist_gradient_boosting,
    train_unsupervised_on_split,
)

__all__ = [
    "DEFAULT_MODEL_PARAMS",
    "DetectionResult",
    "MODEL_SCHEMA_VERSION",
    "META_COLUMNS",
    "TrainingSummary",
    "compute_risk_components",
    "detect_pcap_with_model",
    "summarize_prediction_labels",
    "train_hist_gradient_boosting",
    "train_unsupervised_on_split",
]
