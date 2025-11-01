"""Thin wrappers exposing the canonical unsupervised training helpers."""

from __future__ import annotations

from .predict_unsupervised import (
    DEFAULT_MODEL_PARAMS,
    DetectionResult,
    TrainingSummary,
    detect_pcap_with_model,
    train_unsupervised_on_split,
)

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


__all__ = [
    "DEFAULT_MODEL_PARAMS",
    "DetectionResult",
    "TrainingSummary",
    "META_COLUMNS",
    "detect_pcap_with_model",
    "train_unsupervised_on_split",
]
