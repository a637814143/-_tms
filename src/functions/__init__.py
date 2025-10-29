"""PCAP flow analysis toolkit featuring feature extraction and ML helpers."""

from .modeling import DetectionResult, TrainingSummary, detect_pcap_with_model, train_hist_gradient_boosting
from .static_features import extract_pcap_features, extract_sources_to_jsonl, list_pcap_sources
from .vectorizer import (
    CSVDatasetSummary,
    VectorizationResult,
    load_vectorized_dataset,
    vectorize_jsonl_files,
    vectorize_flows,
    vectorize_pcaps,
)

__all__ = [
    "DetectionResult",
    "CSVDatasetSummary",
    "TrainingSummary",
    "VectorizationResult",
    "detect_pcap_with_model",
    "extract_pcap_features",
    "extract_sources_to_jsonl",
    "list_pcap_sources",
    "load_vectorized_dataset",
    "train_hist_gradient_boosting",
    "vectorize_flows",
    "vectorize_jsonl_files",
    "vectorize_pcaps",
]
