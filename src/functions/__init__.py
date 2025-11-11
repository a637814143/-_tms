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
    "modeling",
    (
        "DetectionResult",
        "TrainingSummary",
        "write_metadata",
        "ModelTrainer",
        "ModelPredictor",
        "detect_pcap_with_model",
        "train_unsupervised_on_split",
        "DEFAULT_MODEL_PARAMS",
        "MODEL_SCHEMA_VERSION",
        "META_COLUMNS",
        "compute_risk_components",
        "summarize_prediction_labels",
    ),
)

_safe_export(
    "feature_extractor",
    (
        "extract_pcap_features",
        "extract_pcap_features_to_file",
        "extract_sources_to_jsonl",
        "list_pcap_sources",
        "extract_features",
        "extract_features_dir",
        "get_loaded_plugin_info",
    ),
)

_safe_export(
    "static_features",
    (
        "extract_pcap_features",
        "extract_pcap_features_to_file",
        "extract_sources_to_jsonl",
        "list_pcap_sources",
        "extract_features",
        "extract_features_dir",
        "get_loaded_plugin_info",
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
        "DataPreprocessor",
        "preprocess_feature_dir",
        "FeatureSource",
    ),
)