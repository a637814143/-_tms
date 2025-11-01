"""Thin wrappers exposing the canonical unsupervised training helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def train_unsupervised_on_split(*args: Any, **kwargs: Any):
    """Proxy to :func:`predict_unsupervised.train_unsupervised_on_split`."""

    from .predict_unsupervised import train_unsupervised_on_split as _impl

    return _impl(*args, **kwargs)


def detect_pcap_with_model(*args: Any, **kwargs: Any):
    """Proxy to :func:`predict_unsupervised.detect_pcap_with_model`."""

    from .predict_unsupervised import detect_pcap_with_model as _impl

    return _impl(*args, **kwargs)


if TYPE_CHECKING:  # pragma: no cover - import only for static analyzers
    from .predict_unsupervised import DetectionResult, TrainingSummary


def __getattr__(name: str) -> Any:  # pragma: no cover - simple lazy proxy
    if name in {"DetectionResult", "TrainingSummary"}:
        from . import predict_unsupervised

        value = getattr(predict_unsupervised, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "DetectionResult",
    "TrainingSummary",
    "META_COLUMNS",
    "detect_pcap_with_model",
    "train_unsupervised_on_split",
]
