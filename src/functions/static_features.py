"""Compatibility layer exposing PCAP feature extraction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

from .feature_extractor import (
    extract_features,
    extract_features_dir,
    extract_pcap_features,
    extract_pcap_features_to_file,
    extract_sources_to_jsonl,
    get_loaded_plugin_info,
    list_pcap_sources,
)

__all__ = [
    "extract_features",
    "extract_features_dir",
    "extract_pcap_features",
    "extract_pcap_features_to_file",
    "extract_sources_to_jsonl",
    "get_loaded_plugin_info",
    "list_pcap_sources",
]

# Provide minimal type aliases so external callers using the legacy module retain
# helpful annotations without needing to import from ``feature_extractor`` directly.
FeatureSourceLike = Union[str, Path, Sequence[Union[str, Path]]]
JSONLSource = Union[str, Path]
PCAPSource = Union[str, Path]
