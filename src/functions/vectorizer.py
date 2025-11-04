"""Utilities for exporting PCAP flow features to CSV and ML matrices."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, overload
from typing import Literal

import csv
import json
import numpy as np

from .static_features import extract_pcap_features

try:  # Optional dependency for command-line progress bars.
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None  # type: ignore


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
CSV_READ_ENCODINGS: Sequence[str] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
_CSV_READ_ENCODINGS = CSV_READ_ENCODINGS


_DUPLICATE_COLUMN_ALIASES: Dict[Tuple[str, int], str] = {
    ("Fwd Header Length", 1): "Fwd Header Length.1",
}


def _resolve_actual_key(column: str, occurrence: int) -> str:
    """Translate a CSV column to the underlying flow dictionary key."""

    return _DUPLICATE_COLUMN_ALIASES.get((column, occurrence), column)


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


def numeric_feature_names() -> List[str]:
    """Return the ordered numeric feature names used for model input."""

    return list(_NUMERIC_FEATURE_NAMES)


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


@dataclass
class LoadedDatasetStats:
    """Metadata describing dataset composition during CSV loading."""

    total_rows: int
    labeled_rows: int
    dropped_rows: int


@dataclass
class LoadedDataset:
    """Container returned when parsing a vectorized CSV dataset."""

    matrix: np.ndarray
    labels: Optional[np.ndarray]
    feature_names: List[str]
    label_mapping: Optional[Dict[int, str]]
    stats: LoadedDatasetStats


def _iter_jsonl_records(path: Union[str, Path]) -> Iterator[Dict[str, object]]:
    """Yield parsed JSON objects from a JSONL file."""

    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:  # pragma: no cover - invalid data
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_number}: {exc}"
                ) from exc


def _iter_flows_from_jsonl(
    inputs: Sequence[Union[str, Path]]
) -> Iterator[Tuple[Dict[str, object], Optional[int]]]:
    """Iterate over all flows contained in one or more JSONL extraction files."""

    for jsonl_path in inputs:
        for record in _iter_jsonl_records(jsonl_path):
            if not record.get("success", False):
                source = record.get("path", str(jsonl_path))
                error = record.get("error", "unknown error")
                raise RuntimeError(f"Extraction failed for {source}: {error}")
            record_label: Optional[int]
            try:
                record_label = int(record.get("label")) if record.get("label") is not None else None
            except (TypeError, ValueError):
                record_label = None
            for flow in record.get("flows", []):
                yield flow, record_label


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


def _create_progress_bar(desc: str, unit: str, show: bool):
    if not show or tqdm is None:
        return None
    return tqdm(desc=desc, unit=unit, leave=False)


def vectorize_pcaps(
    inputs: Sequence[Tuple[Union[str, Path], Optional[int]]],
    output_path: Union[str, Path],
    *,
    show_progress: bool = False,
) -> CSVDatasetSummary:
    """Extract flow features from PCAP files and export them as CSV."""

    path = Path(output_path)
    flow_count = 0
    labels_present = False

    progress = _create_progress_bar("Vectorizing flows", "flow", show_progress)

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
                if progress is not None:
                    progress.update(1)

    if progress is not None:
        progress.close()

    return CSVDatasetSummary(
        path=path,
        flow_count=flow_count,
        column_count=len(CSV_COLUMNS),
        has_labels=labels_present,
    )


def vectorize_jsonl_files(
    inputs: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    *,
    label_override: Optional[int] = None,
    show_progress: bool = False,
) -> CSVDatasetSummary:
    """Convert JSONL extraction results into the mandated CSV format."""

    jsonl_paths = [Path(item) for item in inputs]
    path = Path(output_path)
    flow_count = 0
    labels_present = False

    progress = _create_progress_bar("Vectorizing flows", "flow", show_progress)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)

        for flow, record_label in _iter_flows_from_jsonl(jsonl_paths):
            effective_label = label_override if label_override is not None else record_label
            row = _flow_to_csv_row(flow, effective_label)
            writer.writerow(row)
            flow_count += 1
            if row and row[-1].strip():
                labels_present = True
            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    return CSVDatasetSummary(
        path=path,
        flow_count=flow_count,
        column_count=len(CSV_COLUMNS),
        has_labels=labels_present,
    )


def _load_dataset_from_reader(
    reader: Iterator[List[str]],
    *,
    progress=None,
) -> LoadedDataset:
    """Parse a CSV dataset reader into numeric arrays and label metadata."""

    feature_rows: List[List[float]] = []
    row_labels: List[Optional[str]] = []

    header = next(reader, None)
    if header is None:
        empty = np.zeros((0, len(_NUMERIC_FEATURE_KEYS)), dtype=np.float32)
        stats = LoadedDatasetStats(total_rows=0, labeled_rows=0, dropped_rows=0)
        return LoadedDataset(empty, None, list(_NUMERIC_FEATURE_NAMES), None, stats)

    normalized_header = [column.strip() for column in header]
    expected_header = list(CSV_COLUMNS)
    if normalized_header != expected_header:
        raise ValueError("CSV dataset header does not match the expected format")

    counts: Dict[str, int] = defaultdict(int)
    feature_indices: List[int] = []
    label_index = -1

    for index, column in enumerate(normalized_header):
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
        feature_rows.append(feature_row)

        if label_index >= 0 and label_index < len(row):
            label_value = row[label_index].strip()
            row_labels.append(label_value if label_value else None)
        else:
            row_labels.append(None)

        if progress is not None:
            progress.update(1)

    total_rows = len(feature_rows)
    matrix_rows = feature_rows
    y: Optional[np.ndarray] = None
    label_mapping: Optional[Dict[int, str]] = None

    labeled_rows = 0
    dropped_rows = 0

    if label_index >= 0 and row_labels:
        labeled_pairs = [
            (features, label)
            for features, label in zip(feature_rows, row_labels)
            if label is not None
        ]

        labeled_rows = len(labeled_pairs)
        dropped_rows = total_rows - labeled_rows

        if labeled_pairs:
            matrix_rows = [features for features, _ in labeled_pairs]
            raw_labels = [label for _, label in labeled_pairs]
        else:
            raw_labels = []
    else:
        raw_labels = []

    X = np.asarray(matrix_rows, dtype=np.float32)

    if raw_labels:
        numeric_labels: List[int] = []
        numeric_mapping: Dict[int, str] = {}
        all_numeric = True

        for value in raw_labels:
            try:
                numeric_value = int(value)
            except ValueError:
                all_numeric = False
                break
            numeric_labels.append(numeric_value)
            if numeric_value not in numeric_mapping:
                numeric_mapping[numeric_value] = str(value)

        if all_numeric:
            y = np.asarray(numeric_labels, dtype=np.int64)
            label_mapping = numeric_mapping or None
        else:
            string_to_index: Dict[str, int] = {}
            numeric_labels = []
            for value in raw_labels:
                if value not in string_to_index:
                    string_to_index[value] = len(string_to_index)
                numeric_labels.append(string_to_index[value])
            y = np.asarray(numeric_labels, dtype=np.int64)
            label_mapping = {index: label for label, index in string_to_index.items()}

    if label_mapping is None and y is not None:
        # Preserve numeric class names for downstream reporting.
        unique = dict.fromkeys(int(value) for value in y.tolist())
        label_mapping = {int(value): str(value) for value in unique}

    stats = LoadedDatasetStats(
        total_rows=total_rows,
        labeled_rows=labeled_rows,
        dropped_rows=dropped_rows,
    )

    return LoadedDataset(X, y, list(_NUMERIC_FEATURE_NAMES), label_mapping, stats)


@overload
def load_vectorized_dataset(
    path: Union[str, Path],
    *,
    show_progress: bool = ...,
    return_stats: Literal[False] = ...,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    List[str],
    Optional[Dict[int, str]],
]:
    ...


@overload
def load_vectorized_dataset(
    path: Union[str, Path],
    *,
    show_progress: bool = ...,
    return_stats: Literal[True],
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    List[str],
    Optional[Dict[int, str]],
    LoadedDatasetStats,
]:
    ...


def load_vectorized_dataset(
    path: Union[str, Path],
    *,
    show_progress: bool = False,
    return_stats: bool = False,
) -> Union[
    Tuple[
        np.ndarray,
        Optional[np.ndarray],
        List[str],
        Optional[Dict[int, str]],
    ],
    Tuple[
        np.ndarray,
        Optional[np.ndarray],
        List[str],
        Optional[Dict[int, str]],
        LoadedDatasetStats,
    ],
]:
    """Load a CSV dataset created by :func:`vectorize_pcaps`."""

    dataset_path = Path(path)
    last_error: Optional[UnicodeDecodeError] = None

    for encoding in _CSV_READ_ENCODINGS:
        progress = None
        try:
            with dataset_path.open("r", newline="", encoding=encoding) as handle:
                reader = csv.reader(handle)
                progress = _create_progress_bar("Loading dataset", "row", show_progress)
                loaded = _load_dataset_from_reader(reader, progress=progress)
                if return_stats:
                    return (
                        loaded.matrix,
                        loaded.labels,
                        loaded.feature_names,
                        loaded.label_mapping,
                        loaded.stats,
                    )
                return (
                    loaded.matrix,
                    loaded.labels,
                    loaded.feature_names,
                    loaded.label_mapping,
                )
        except UnicodeDecodeError as exc:  # pragma: no cover - environment specific
            last_error = exc
            continue
        finally:
            if progress is not None:
                progress.close()

    raise ValueError(
        "Unable to decode CSV dataset using supported encodings: "
        + ", ".join(_CSV_READ_ENCODINGS)
    )