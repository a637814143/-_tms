"""Utilities for exporting PCAP flow features to CSV and ML matrices."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)
from typing import Literal

import csv
import glob
import json
import os
import numpy as np

from .csv_utils import read_csv_flexible
from .feature_extractor import extract_pcap_features

try:  # Optional dependency for command-line progress bars.
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None  # type: ignore

try:  # pandas 在运行环境中是可选的
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas 可能未安装
    pd = None  # type: ignore


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
_OPTIONAL_COLUMNS = {"LabelBinary"}

# 兼容 UNSW 风格的表头：使用归一化后的键做映射，避免列名不符导致的数据丢失。
_COLUMN_ALIASES: Dict[str, str] = {
    "srcip": "Source IP",
    "saddr": "Source IP",
    "dstip": "Destination IP",
    "daddr": "Destination IP",
    "sport": "Source Port",
    "srcport": "Source Port",
    "spt": "Source Port",
    "dsport": "Destination Port",
    "dport": "Destination Port",
    "proto": "Protocol",
    "protocol": "Protocol",
    "dur": "Flow Duration",
    "duration": "Flow Duration",
    "flowduration": "Flow Duration",
    "spkts": "Total Fwd Packets",
    "dpkts": "Total Backward Packets",
    "sbytes": "Total Length of Fwd Packets",
    "dbytes": "Total Length of Bwd Packets",
}

ProgressCallback = Optional[Callable[[int], None]]
FeatureSource = Union[str, Sequence[str]]
_DATASET_META_TYPE = "merged_feature_dataset"


_DUPLICATE_COLUMN_ALIASES: Dict[Tuple[str, int], str] = {
    ("Fwd Header Length", 1): "Fwd Header Length.1",
}


def _normalise_header_name(name: str) -> str:
    """归一化列名：小写、去除空格/下划线/连字符，便于宽松匹配。"""

    lowered = name.strip().lower()
    return "".join(ch for ch in lowered if ch.isalnum())


def _apply_header_aliases(columns: Sequence[str]) -> List[str]:
    """将常见 UNSW 风格表头映射到内部统一名称。"""

    resolved: List[str] = []
    for name in columns:
        normalised = _normalise_header_name(str(name))
        resolved.append(_COLUMN_ALIASES.get(normalised, str(name).strip()))
    return resolved


def available_feature_keys(flows: Iterable[Dict[str, object]]) -> List[str]:
    """Collect the union of feature keys that are present in flow dictionaries."""

    keys: set[str] = set()
    for flow in flows:
        for key, value in flow.items():
            if value is None:
                continue
            keys.add(str(key))
    return sorted(keys)


def common_feature_subset(
    model_features: Sequence[str], available_features: Sequence[str]
) -> List[str]:
    """根据模型特征和当前可用特征求交集，保持模型特征顺序。"""

    available = set(available_features)
    return [name for name in model_features if name in available]


def _notify(cb: ProgressCallback, value: int) -> None:
    if cb:
        cb(max(0, min(100, int(value))))


def _resolve_feature_sources(feature_dir: FeatureSource) -> Tuple[List[str], Optional[str]]:
    csv_files: List[str] = []
    resolved_source: Optional[str] = None

    if isinstance(feature_dir, (list, tuple, set)):
        for entry in feature_dir:
            if not isinstance(entry, str):
                continue
            path = os.path.abspath(entry)
            if os.path.isfile(path):
                csv_files.append(path)
        if not csv_files:
            raise RuntimeError("没有选择任何有效的特征 CSV 文件。")
        try:
            resolved_source = os.path.commonpath(csv_files)
        except ValueError:
            resolved_source = os.path.dirname(csv_files[0])
    else:
        resolved = os.path.abspath(str(feature_dir))
        if os.path.isdir(resolved):
            patterns = ["*.csv", "*.CSV"]
            for pattern in patterns:
                csv_files.extend(glob.glob(os.path.join(resolved, pattern)))
            csv_files = sorted(set(csv_files))
            resolved_source = resolved
        elif os.path.isfile(resolved):
            csv_files = [resolved]
            resolved_source = os.path.dirname(resolved)
        else:
            raise FileNotFoundError(f"未找到特征数据来源: {feature_dir}")

    if not csv_files:
        raise RuntimeError("未能在所选路径中找到特征 CSV 文件。")

    return csv_files, resolved_source


def _align_dataframe(frame: "pd.DataFrame") -> "pd.DataFrame":
    header = list(CSV_COLUMNS)
    df = frame.copy()
    df.columns = _apply_header_aliases(df.columns)

    for column in header:
        if column not in df.columns:
            if column in _STRING_COLUMNS or column == _LABEL_COLUMN:
                df[column] = ""
            else:
                df[column] = 0.0

    for column in header:
        if column in _STRING_COLUMNS:
            df[column] = df[column].fillna("").astype(str)
        elif column == _LABEL_COLUMN:
            df[column] = df[column].where(df[column].notna(), "").astype(str)
        else:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    return df.loc[:, header]


def _encode_label_binary(df: "pd.DataFrame") -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("缺少 pandas 依赖，无法执行标签编码。")
    if _LABEL_COLUMN not in df.columns:
        return df

    text = df[_LABEL_COLUMN].astype(str).str.upper()
    df["LabelBinary"] = (~text.str.contains("BENIGN")).astype(int)
    return df


def build_clean_training_dataset(
    input_csv: Union[str, Path],
    output_csv: Union[str, Path],
    *,
    feature_columns: Optional[Sequence[str]] = None,
    include_label_binary: bool = True,
) -> Path:
    """Clean a raw CIC flow CSV into a numeric training-ready dataset.

    The routine closely follows the explicit pandas workflow shared in the
    user instructions: strip column whitespace, retain only the ordered numeric
    features and ``Label`` column, derive a binary label, coerce features to
    numeric values (filling invalid entries with ``0``), and write the result to
    ``output_csv``.
    """

    if pd is None:  # pragma: no cover - pandas may be optional at runtime
        raise RuntimeError("缺少 pandas 依赖，无法清洗原始数据集。")

    features = list(feature_columns) if feature_columns is not None else numeric_feature_names()

    raw = pd.read_csv(input_csv)
    raw.columns = [str(column).strip() for column in raw.columns]

    if _LABEL_COLUMN not in raw.columns:
        raise ValueError("原始数据集中缺少 Label 列，无法生成训练集。")

    cleaned = raw.copy()
    for column in features:
        if column not in cleaned.columns:
            cleaned[column] = 0.0

    cleaned = cleaned.loc[:, features + [_LABEL_COLUMN]]

    if include_label_binary:
        text = cleaned[_LABEL_COLUMN].astype(str).str.upper()
        cleaned["LabelBinary"] = (~text.str.contains("BENIGN")).astype(int)

    cleaned[features] = cleaned[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return output_path


def _normalise_csv_record(record: Dict[str, object], header: Sequence[str]) -> Tuple[List[object], bool]:
    """Normalise a CSV row without pandas and report whether it is labelled."""

    aligned: List[object] = []
    has_label = False

    for column in header:
        value = record.get(column)
        if column in _STRING_COLUMNS:
            aligned.append("" if value is None else str(value))
        elif column == _LABEL_COLUMN:
            label_value = "" if value is None else str(value)
            has_label = bool(label_value.strip())
            aligned.append(label_value)
        else:
            try:
                aligned.append(float(value))
            except (TypeError, ValueError):
                aligned.append(0.0)

    return aligned, has_label


def _append_csv_file_without_pandas(
    csv_path: str,
    dataset_path: str,
    header: Sequence[str],
) -> Tuple[int, int]:
    """Append normalised CSV rows to the dataset when pandas is absent."""

    encoding_candidates = list(CSV_READ_ENCODINGS)
    if None not in encoding_candidates:
        encoding_candidates.append(None)

    last_unicode_error: Optional[UnicodeDecodeError] = None
    tried_encodings: List[str] = []

    for encoding in encoding_candidates:
        open_kwargs = {"newline": ""}
        if encoding is not None:
            open_kwargs["encoding"] = encoding

        try:
            with open(csv_path, **open_kwargs) as source_handle:
                reader = csv.DictReader(source_handle)
                if reader.fieldnames is None:
                    return 0, 0

                rows = 0
                labeled_rows = 0
                with open(dataset_path, "a", encoding="utf-8", newline="") as target_handle:
                    writer = csv.writer(target_handle)
                    for raw_row in reader:
                        if not raw_row:
                            continue
                        raw_row.pop(None, None)
                        aligned_row, has_label = _normalise_csv_record(raw_row, header)
                        writer.writerow(aligned_row)
                        rows += 1
                        if has_label:
                            labeled_rows += 1

                return rows, labeled_rows
        except UnicodeDecodeError as exc:
            tried_encodings.append(encoding or "<default>")
            last_unicode_error = exc
            continue

    if last_unicode_error is not None:
        detail = ", ".join(tried_encodings) or "未知编码"
        raise UnicodeDecodeError(
            last_unicode_error.encoding or "utf-8",
            last_unicode_error.object,
            last_unicode_error.start,
            last_unicode_error.end,
            f"无法读取 CSV（尝试编码: {detail}）",
        ) from last_unicode_error

    raise RuntimeError(f"无法读取 CSV 文件: {csv_path}")


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
    missing_feature_counts: List[int]
    available_feature_counts: List[int]
    coverage_ratio: float

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
    missing_feature_counts: List[int] = []
    available_feature_counts: List[int] = []

    for row_index, flow in enumerate(flow_list):
        missing_count = 0
        available_count = 0
        for col_index, key in enumerate(feature_keys):
            value = flow.get(key, 0.0)
            present = key in flow and flow.get(key) is not None
            try:
                matrix[row_index, col_index] = float(value)
            except (TypeError, ValueError):
                matrix[row_index, col_index] = 0.0
                present = False

            if present:
                available_count += 1
            else:
                missing_count += 1

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

        missing_feature_counts.append(missing_count)
        available_feature_counts.append(available_count)

    labels: Optional[np.ndarray]
    if has_missing_labels:
        labels = None
    else:
        labels = np.asarray(label_values, dtype=np.int64)
        if labels.size != matrix.shape[0]:
            labels = None

    coverage_ratio = 0.0
    if feature_keys:
        coverage_values = [count / len(feature_keys) for count in available_feature_counts]
        coverage_ratio = float(np.mean(coverage_values)) if coverage_values else 0.0

    return VectorizationResult(
        matrix=matrix,
        labels=labels,
        feature_names=list(feature_keys),
        missing_feature_counts=missing_feature_counts,
        available_feature_counts=available_feature_counts,
        coverage_ratio=coverage_ratio,
    )


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


class DataPreprocessor:
    """高层封装：把原始特征 CSV 清洗成  数值特征 + Label(+LabelBinary) 的训练数据集。"""

    def __init__(
        self,
        *,
        feature_columns: Sequence[str] = None,
        include_label_binary: bool = True,
    ):
        # 默认：所有数值特征 + Label
        if feature_columns is None:
            numeric_cols = numeric_feature_names()
            self.feature_columns = list(numeric_cols) + [_LABEL_COLUMN]
        else:
            self.feature_columns = list(feature_columns)

        self.include_label_binary = include_label_binary

    def clean_data(self, frame: "pd.DataFrame") -> "pd.DataFrame":
        """只做两件事：
        1）把需要的数值特征全部转成 float，缺失的补 0；
        2）保证有 Label 列，按需要生成 LabelBinary。
        """
        if pd is None:
            raise RuntimeError("缺少 pandas 依赖，无法执行 DataFrame 清洗。")

        if frame is None or frame.empty:
            raise ValueError("输入的 DataFrame 为空，无法进行数据预处理。")

        df = frame.copy()

        # 1. 统一列名：去掉首尾空格，并套用别名，避免 ' Flow Duration ' 以及 UNSW 风格的缩写问题
        df.columns = _apply_header_aliases([str(col).strip() for col in df.columns])

        # 2. 确保有 Label 列（原始 CICIDS 里就是 Label，有些文件可能有空格/大小写问题）
        if _LABEL_COLUMN not in df.columns:
            for cand in (" Label", "label", "LABEL"):
                if cand in df.columns:
                    df[_LABEL_COLUMN] = df[cand].astype(str)
                    break
            else:
                # 实在没有就补一列空字符串，后续当成“无标签”
                df[_LABEL_COLUMN] = ""

        # 3. 为每个数值特征准备一列，并强制转成 float，非法值/缺失值 → 0.0
        for col in numeric_feature_names():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                df[col] = 0.0

        # 4. 需要的话，生成二进制标签列：BENIGN -> 0，其它 -> 1
        if self.include_label_binary:
            df = _encode_label_binary(df)

        return df

    def select_features(self, frame: "pd.DataFrame") -> "pd.DataFrame":
        columns = list(self.feature_columns)
        if self.include_label_binary and "LabelBinary" in frame.columns:
            if "LabelBinary" not in columns:
                columns.append("LabelBinary")
        return frame.loc[:, columns]

    def vectorize(self, frame: "pd.DataFrame") -> "pd.DataFrame":
        cleaned = self.clean_data(frame)
        return self.select_features(cleaned)

    def preprocess(
        self,
        feature_dir: FeatureSource,
        output_dir: str,
        *,
        progress_cb: ProgressCallback = None,
    ) -> Dict[str, object]:
        csv_files, resolved_source = _resolve_feature_sources(feature_dir)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"dataset_{timestamp}"
        dataset_path = os.path.join(output_dir, f"{base_name}.csv")
        manifest_path = os.path.join(output_dir, f"{base_name}_manifest.csv")
        meta_path = os.path.join(output_dir, f"{base_name}_meta.json")

        # 如果文件名已存在，就在后面加 _1、_2……
        counter = 1
        while os.path.exists(dataset_path):
            base_name = f"dataset_{timestamp}_{counter}"
            dataset_path = os.path.join(output_dir, f"{base_name}.csv")
            manifest_path = os.path.join(output_dir, f"{base_name}_manifest.csv")
            meta_path = os.path.join(output_dir, f"{base_name}_meta.json")
            counter += 1

        header = list(self.feature_columns)
        if self.include_label_binary and "LabelBinary" not in header:
            header.append("LabelBinary")

        manifest_rows: List[Dict[str, object]] = []
        total_rows = 0
        labeled_rows = 0

        # 先写表头
        with open(dataset_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)

        total_files = len(csv_files)
        use_pandas = pd is not None

        for index, csv_path in enumerate(csv_files, start=1):
            if progress_cb:
                _notify(progress_cb, int((index - 1) / total_files * 100))

            if use_pandas:
                df = read_csv_flexible(csv_path)
                aligned = self.vectorize(df)
                aligned.to_csv(
                    dataset_path,
                    mode="a",
                    header=False,
                    index=False,
                    encoding="utf-8",
                )

                rows = int(aligned.shape[0])
                if rows:
                    if "LabelBinary" in aligned.columns:
                        # LabelBinary 为 0/1 的行视为“有标签”
                        labeled = int(aligned["LabelBinary"].isin([0, 1]).sum())
                    else:
                        label_series = aligned[_LABEL_COLUMN].astype(str)
                        labeled = int(label_series.str.strip().ne("").sum())
                else:
                    labeled = 0
            else:  # pragma: no cover - 仅在缺少 pandas 时运行
                rows, labeled = _append_csv_file_without_pandas(csv_path, dataset_path, header)

            manifest_rows.append(
                {
                    "source_file": os.path.basename(csv_path),
                    "source_path": os.path.abspath(csv_path),
                    "rows": rows,
                    "labeled_rows": labeled,
                }
            )
            total_rows += rows
            labeled_rows += labeled

            if total_files:
                _notify(progress_cb, int(index / total_files * 100))

        # 写 manifest
        with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
            fieldnames = ["source_file", "source_path", "rows", "labeled_rows"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

        # 写 meta.json
        meta_payload = {
            "type": _DATASET_META_TYPE,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_feature_dir": os.path.abspath(resolved_source) if resolved_source else "",
            "files": [os.path.abspath(path) for path in csv_files],
            "columns": header,
            "string_columns": sorted(_STRING_COLUMNS),
            "label_column": _LABEL_COLUMN,
            "rows": int(total_rows),
            "labeled_rows": int(labeled_rows),
            "unlabeled_rows": int(total_rows - labeled_rows),
            "dataset_format": "csv",
            "preprocessor": {
                "feature_order": header,
                "input_columns": header,
            },
        }

        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta_payload, handle, ensure_ascii=False, indent=2)

        _notify(progress_cb, 100)

        return {
            "dataset_path": dataset_path,
            "manifest_path": manifest_path,
            "meta_path": meta_path,
            "total_rows": int(total_rows),
            "total_cols": len(header) - 1,  # 排除 Label 列
            "feature_columns": header,
            "files": csv_files,
        }


def preprocess_feature_dir(
    feature_dir: FeatureSource,
    output_dir: str,
    *,
    progress_cb: ProgressCallback = None,
) -> Dict[str, object]:
    """兼容旧接口的包装函数。"""

    processor = DataPreprocessor()
    return processor.preprocess(feature_dir, output_dir, progress_cb=progress_cb)


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
    numeric_header = list(_NUMERIC_FEATURE_NAMES)
    valid_headers = {
        tuple(expected_header),
        tuple(expected_header + ["LabelBinary"]),
        tuple(numeric_header + [_LABEL_COLUMN]),
        tuple(numeric_header + [_LABEL_COLUMN, "LabelBinary"]),
    }
    if tuple(normalized_header) not in valid_headers:
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
        if column in _OPTIONAL_COLUMNS:
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

        # --- 修正标签读取逻辑：优先使用行尾二进制标签 ---
        # 如果这一行的长度比表头多，且最后一个值看起来是 0/1，
        # 说明我们遇到了像 dataset_20251126_*.csv 这种“行尾多出 LabelBinary”的情况。
        if len(row) > len(header) and row[-1].strip() in ("0", "1"):
            # 用最后一个值作为标签（实际就是 LabelBinary）
            label_value = row[-1].strip()
            row_labels.append(label_value)
        elif label_index >= 0 and label_index < len(row):
            # 正常情况：表头里的 "Label" 列就是标签
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