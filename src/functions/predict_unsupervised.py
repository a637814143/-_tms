"""Compatibility facade for legacy unsupervised training imports."""

from __future__ import annotations

from .unsupervised_train import (
    DEFAULT_MODEL_PARAMS,
    DetectionResult,
    TrainingSummary,
    detect_pcap_with_model,
    train_unsupervised_on_split,
)

__all__ = [
    "DEFAULT_MODEL_PARAMS",
    "DetectionResult",
    "TrainingSummary",
    "detect_pcap_with_model",
    "train_unsupervised_on_split",
]
