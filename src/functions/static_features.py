
"""Backwards-compatible wrappers around :mod:`feature_extractor` helpers."""

from __future__ import annotations

from .feature_extractor import (
    extract_pcap_features,
    extract_pcap_features_batch,
    extract_pcap_features_to_file,
    extract_sources_to_jsonl,
    list_pcap_sources,
)

__all__ = [
    "extract_pcap_features",
    "extract_pcap_features_batch",
    "extract_pcap_features_to_file",
    "extract_sources_to_jsonl",
    "list_pcap_sources",
]
