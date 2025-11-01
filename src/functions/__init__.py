"""Helper exports for the :mod:`src.functions` package."""

from __future__ import annotations

from importlib import import_module
from typing import Sequence

__all__: list[str] = []


def _safe_export(module_name: str, symbols: Sequence[str]) -> None:
    """Attempt to import ``module_name`` and re-export ``symbols`` if available."""

    try:
        module = import_module(f"{__name__}.{module_name}")
    except Exception:
        return

    for symbol in symbols:
        if hasattr(module, symbol):
            globals()[symbol] = getattr(module, symbol)
            __all__.append(symbol)


_safe_export(
    "unsupervised_train",
    (
        "DetectionResult",
        "TrainingSummary",
        "detect_pcap_with_model",
        "train_unsupervised_on_split",
        "DEFAULT_MODEL_PARAMS",
        "META_COLUMNS",
    ),
)

_safe_export(
    "static_features",
    (
        "extract_pcap_features",
        "extract_sources_to_jsonl",
        "list_pcap_sources",
    ),
)

_safe_export(
    "vectorizer",
    (
        "CSVDatasetSummary",
        "VectorizationResult",
        "load_vectorized_dataset",
        "vectorize_jsonl_files",
        "vectorize_flows",
        "vectorize_pcaps",
    ),
)