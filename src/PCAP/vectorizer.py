"""Utilities for exporting PCAP flow features to CSV and ML matrices."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import csv
import numpy as np

from .static_features import extract_pcap_features


# Ordered CSV header mandated by the downstream pipeline.
CSV_COLUMNS: Sequence[str] = (
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Label",
)

_STRING_COLUMNS = {"Flow ID", "Source IP", "Destination IP", "Timestamp"}
_LABEL_COLUMN = "Label"


def _resolve_actual_key(column: str, occurrence: int) -> str:
    """Translate a CSV column to the underlying flow dictionary key."""

    if column == "Fwd Header Length" and occurrence == 1:
        return "Fwd Header Length.1"
    return column


def _numeric_feature_metadata() -> List[str]:
    """Return the ordered keys used for numeric model input."""

    feature_keys: List[str] = []
    counts: Dict[str, int] = defaultdict(int)

    for column in CSV_COLUMNS:
        occurrence = counts[column]
        counts[column] += 1

        if column in _STRING_COLUMNS or column == _LABEL_COLUMN:
            continue

        actual_key = _resolve_actual_key(column, occurrence)
        feature_keys.append(actual_key)

    return feature_keys


_NUMERIC_FEATURE_KEYS = _numeric_feature_metadata()
_NUMERIC_FEATURE_NAMES = list(_NUMERIC_FEATURE_KEYS)


@dataclass
class VectorizationResult:
    """Container describing the numeric representation of flow records."""

    matrix: np.ndarray
    labels: Optional[np.ndarray]
    feature_names: List[str]

    @property
    def flow_count(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.matrix.shape[1])


@dataclass
class CSVDatasetSummary:
    """Summary describing a CSV export produced from PCAP flows."""

    path: Path
    flow_count: int
    column_count: int
    has_labels: bool


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, int, np.integer, np.floating)):
        return str(value)
    return str(value)


def _flow_to_csv_row(flow: Dict[str, object], label_override: Optional[int]) -> List[str]:
    row: List[str] = []
    counts: Dict[str, int] = defaultdict(int)

    for column in CSV_COLUMNS:
        occurrence = counts[column]
        counts[column] += 1

        if column == _LABEL_COLUMN:
            value = label_override if label_override is not None else flow.get("Label")
        else:
            key = _resolve_actual_key(column, occurrence)
            value = flow.get(key)

        row.append(_format_value(value))

    return row


def vectorize_flows(
    flows: Iterable[Dict[str, object]],
    *,
    feature_names: Optional[Sequence[str]] = None,
    default_label: Optional[int] = None,
    dtype: np.dtype = np.float32,
    include_labels: bool = True,
) -> VectorizationResult:
    """Convert flow dictionaries into a numeric matrix for modeling."""

    flow_list = [dict(flow) for flow in flows]
    feature_keys = list(feature_names) if feature_names is not None else list(_NUMERIC_FEATURE_KEYS)
    matrix = np.zeros((len(flow_list), len(feature_keys)), dtype=dtype)
    label_values: List[int] = []
    has_missing_labels = not include_labels

    for row_index, flow in enumerate(flow_list):
        for col_index, key in enumerate(feature_keys):
            value = flow.get(key, 0.0)
            try:
                matrix[row_index, col_index] = float(value)
            except (TypeError, ValueError):
                matrix[row_index, col_index] = 0.0

        if include_labels:
            label = flow.get("Label")
            if label is None and default_label is not None:
                label = default_label
                flow["Label"] = default_label

            if label is None:
                has_missing_labels = True
            else:
                try:
                    label_values.append(int(label))
                except (TypeError, ValueError):
                    has_missing_labels = True

    labels: Optional[np.ndarray]
    if has_missing_labels:
        labels = None
    else:
        labels = np.asarray(label_values, dtype=np.int64)
        if labels.size != matrix.shape[0]:
            labels = None

    return VectorizationResult(matrix=matrix, labels=labels, feature_names=list(feature_keys))


def vectorize_pcaps(
    inputs: Sequence[Tuple[Union[str, Path], Optional[int]]],
    output_path: Union[str, Path],
) -> CSVDatasetSummary:
    """Extract flow features from PCAP files and export them as CSV."""

    path = Path(output_path)
    flow_count = 0
    labels_present = False

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)

        for pcap_path, label in inputs:
            result = extract_pcap_features(pcap_path)
            if not result.get("success", False):
                raise RuntimeError(
                    f"Failed to extract features from {pcap_path}: {result.get('error', 'unknown error')}"
                )

            for flow in result.get("flows", []):
                row = _flow_to_csv_row(flow, label)
                writer.writerow(row)
                flow_count += 1
                if len(row) >= 1 and row[-1].strip() != "":
                    labels_present = True

    return CSVDatasetSummary(
        path=path,
        flow_count=flow_count,
        column_count=len(CSV_COLUMNS),
        has_labels=labels_present,
    )


def load_vectorized_dataset(path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Load a CSV dataset created by :func:`vectorize_pcaps`."""

    matrix: List[List[float]] = []
    labels: List[int] = []

    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return np.zeros((0, len(_NUMERIC_FEATURE_KEYS)), dtype=np.float32), None, list(
                _NUMERIC_FEATURE_NAMES
            )

        expected_header = list(CSV_COLUMNS)
        if list(header) != expected_header:
            raise ValueError("CSV dataset header does not match the expected format")

        counts: Dict[str, int] = defaultdict(int)
        feature_indices: List[int] = []
        label_index = -1

        for index, column in enumerate(header):
            occurrence = counts[column]
            counts[column] += 1
            if column == _LABEL_COLUMN:
                label_index = index
                continue
            if column in _STRING_COLUMNS:
                continue
            feature_indices.append(index)

        if len(feature_indices) != len(_NUMERIC_FEATURE_KEYS):
            raise ValueError("CSV dataset feature count mismatch")

        for row in reader:
            if not row:
                continue
            feature_row: List[float] = []
            for index in feature_indices:
                value = row[index].strip()
                try:
                    feature_row.append(float(value) if value else 0.0)
                except ValueError:
                    feature_row.append(0.0)
            matrix.append(feature_row)

            if label_index >= 0 and label_index < len(row):
                label_value = row[label_index].strip()
                if label_value:
                    labels.append(int(label_value))

    X = np.asarray(matrix, dtype=np.float32)
    y: Optional[np.ndarray]
    if labels and len(labels) == len(matrix):
        y = np.asarray(labels, dtype=np.int64)
    else:
        y = None

    return X, y, list(_NUMERIC_FEATURE_NAMES)
