"""Compatibility layer re-exporting vectorization helpers."""

from __future__ import annotations

from .vectorizer import (
    CSVDatasetSummary,
    LoadedDataset,
    LoadedDatasetStats,
    VectorizationResult,
    CSV_COLUMNS,
    load_vectorized_dataset,
    vectorize_flows,
    vectorize_jsonl_files,
    vectorize_pcaps,
)

__all__ = [
    "CSVDatasetSummary",
    "LoadedDataset",
    "LoadedDatasetStats",
    "VectorizationResult",
    "CSV_COLUMNS",
    "load_vectorized_dataset",
    "vectorize_flows",
    "vectorize_jsonl_files",
    "vectorize_pcaps",
]