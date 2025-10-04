"""特征 CSV 数据预处理，将其整理成可直接用于模型训练的标准化数据集。"""

import glob
import json
import math
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

ProgressCallback = Optional[Callable[[int], None]]

# 在预处理后仍保留的原始元信息列，用于训练时按文件等维度汇总
RESERVED_META_COLUMNS = [
    "__source_file__",
    "__source_path__",
    "pcap_file",
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
]

MISSING_TOKEN = "<MISSING>"


def _notify(cb: ProgressCallback, value: int) -> None:
    if cb:
        cb(max(0, min(100, int(value))))


def _load_feature_csv(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="gbk")
    if df.empty:
        raise ValueError(f"CSV 没有任何数据: {csv_path}")
    return df


def _median_or_default(series: pd.Series) -> float:
    non_na = series.dropna()
    if non_na.empty:
        return 0.0
    median = float(non_na.median())
    if math.isnan(median) or math.isinf(median):
        median = float(non_na.iloc[0])
    return median


def _update_numeric_stats(name: str, values: pd.Series, store: Dict[str, Dict[str, float]]) -> None:
    if values.empty:
        return
    arr = np.asarray(values, dtype=np.float64)
    stats = store.setdefault(
        name,
        {"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "min": None, "max": None},
    )
    count = float(arr.size)
    stats["count"] += count
    stats["sum"] += float(arr.sum())
    stats["sum_sq"] += float(np.multiply(arr, arr).sum())
    current_min = float(arr.min())
    current_max = float(arr.max())
    stats["min"] = current_min if stats["min"] is None else min(stats["min"], current_min)
    stats["max"] = current_max if stats["max"] is None else max(stats["max"], current_max)


def _process_single_dataframe(
    df: pd.DataFrame,
    *,
    fill_strategies: Dict[str, float],
    categorical_maps: Dict[str, Dict[str, object]],
    numeric_stats: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """处理单个特征表，返回预处理后的 DataFrame、元信息列和特征列顺序。"""

    meta_cols = [col for col in RESERVED_META_COLUMNS if col in df.columns]
    feature_columns: List[str] = []
    processed_data: Dict[str, pd.Series] = {}

    for column in df.columns:
        if column in meta_cols:
            continue

        series = df[column]

        if is_bool_dtype(series):
            encoded = series.fillna(False).astype(bool).astype("int8")
            fill_strategies.setdefault(column, 0.0)
            _update_numeric_stats(column, encoded, numeric_stats)
            processed_data[column] = encoded
            feature_columns.append(column)
        elif is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if column not in fill_strategies:
                fill_strategies[column] = _median_or_default(numeric)
            filled = numeric.fillna(fill_strategies[column]).astype("float32")
            _update_numeric_stats(column, filled, numeric_stats)
            processed_data[column] = filled
            feature_columns.append(column)
        else:
            filled = series.fillna(MISSING_TOKEN)
            if is_datetime64_any_dtype(filled):
                filled = filled.astype(str)
            elif not is_categorical_dtype(filled):
                filled = filled.astype(str)
            normalized = filled.str.strip().replace("", MISSING_TOKEN)

            encoded_name = f"{column}__code"
            entry = categorical_maps.setdefault(
                encoded_name,
                {
                    "source_column": column,
                    "labels": [],
                    "missing_token": MISSING_TOKEN,
                    "mapping": {},
                },
            )
            mapping = entry["mapping"]
            codes = normalized.map(mapping)
            missing_mask = codes.isna()
            if missing_mask.any():
                for value in normalized[missing_mask].unique():
                    label = str(value)
                    if label not in mapping:
                        mapping[label] = len(entry["labels"])
                        entry["labels"].append(label)
                codes = normalized.map(mapping)
            encoded = codes.fillna(-1).astype("int32")
            processed_data[encoded_name] = encoded
            feature_columns.append(encoded_name)

    if not processed_data:
        raise RuntimeError("没有找到可用于训练的特征列。")

    parts: List[pd.DataFrame] = []
    if meta_cols:
        parts.append(df.loc[:, meta_cols])
    parts.append(pd.DataFrame(processed_data, index=df.index))
    processed_df = pd.concat(parts, axis=1)
    return processed_df, meta_cols, feature_columns


FeatureSource = Union[str, Sequence[str]]


def preprocess_feature_dir(
    feature_dir: FeatureSource,
    output_dir: str,
    *,
    progress_cb: ProgressCallback = None,
) -> Dict[str, object]:
    """批量读取特征 CSV，执行数据预处理并输出统一的训练数据集。"""

    resolved_source: Union[str, None] = None
    csv_files: List[str] = []

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

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"dataset_preprocessed_{timestamp}"
    dataset_path = os.path.join(output_dir, f"{base_name}.csv")
    manifest_path = os.path.join(output_dir, f"{base_name}_manifest.csv")
    meta_path = os.path.join(output_dir, f"{base_name}_meta.json")

    counter = 1
    while os.path.exists(dataset_path):
        base_name = f"dataset_preprocessed_{timestamp}_{counter}"
        dataset_path = os.path.join(output_dir, f"{base_name}.csv")
        manifest_path = os.path.join(output_dir, f"{base_name}_manifest.csv")
        meta_path = os.path.join(output_dir, f"{base_name}_meta.json")
        counter += 1

    fill_strategies: Dict[str, float] = {}
    categorical_maps: Dict[str, Dict[str, object]] = {}
    numeric_stats: Dict[str, Dict[str, float]] = {}

    manifest_rows: List[Dict[str, object]] = []
    total_rows = 0
    total_files = len(csv_files)
    meta_columns_global: List[str] = []
    feature_columns_global: List[str] = []
    first_chunk = True

    for idx, csv_path in enumerate(csv_files, start=1):
        df = _load_feature_csv(csv_path)
        df = df.copy()
        df["__source_file__"] = os.path.basename(csv_path)
        df["__source_path__"] = os.path.abspath(csv_path)

        processed_df, meta_cols, feature_columns = _process_single_dataframe(
            df,
            fill_strategies=fill_strategies,
            categorical_maps=categorical_maps,
            numeric_stats=numeric_stats,
        )

        if first_chunk:
            meta_columns_global = meta_cols
            feature_columns_global = feature_columns
        else:
            if meta_cols != meta_columns_global:
                raise RuntimeError(
                    f"文件 {os.path.basename(csv_path)} 的元信息列与之前不一致。"
                )
            if feature_columns != feature_columns_global:
                missing = [c for c in feature_columns_global if c not in feature_columns]
                extra = [c for c in feature_columns if c not in feature_columns_global]
                if missing or extra:
                    raise RuntimeError(
                        f"文件 {os.path.basename(csv_path)} 的特征列与之前不一致。"
                    )

        full_column_order = list(meta_columns_global) + feature_columns_global
        processed_df = processed_df.loc[:, full_column_order]

        processed_df.to_csv(
            dataset_path,
            index=False,
            encoding="utf-8",
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )

        rows = int(len(processed_df))
        manifest_rows.append(
            {
                "source_file": os.path.basename(csv_path),
                "source_path": os.path.abspath(csv_path),
                "start_index": total_rows,
                "end_index": total_rows + rows,
                "rows": rows,
            }
        )
        total_rows += rows
        first_chunk = False

        _notify(progress_cb, 10 + int(80 * idx / total_files))

    if first_chunk:
        raise RuntimeError("未能生成任何预处理数据。")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")

    feature_columns = feature_columns_global

    column_profiles: List[Dict[str, object]] = []
    for column in feature_columns:
        if column in numeric_stats:
            stats = numeric_stats[column]
            count = stats["count"] or 0.0
            if count:
                mean = stats["sum"] / count
                variance = max(stats["sum_sq"] / count - mean * mean, 0.0)
                std = math.sqrt(variance)
            else:
                mean = 0.0
                std = 0.0
            column_profiles.append(
                {
                    "name": column,
                    "kind": "numeric",
                    "min": float(stats["min"] if stats["min"] is not None else 0.0),
                    "max": float(stats["max"] if stats["max"] is not None else 0.0),
                    "mean": float(mean),
                    "std": float(std),
                }
            )
        else:
            meta = categorical_maps.get(column, {})
            column_profiles.append(
                {
                    "name": column,
                    "kind": "categorical_encoded",
                    "source_column": meta.get("source_column"),
                    "unique": len(meta.get("labels", [])),
                }
            )

    categorical_serializable = {}
    for key, meta in categorical_maps.items():
        categorical_serializable[key] = {
            "source_column": meta.get("source_column"),
            "labels": list(meta.get("labels", [])),
            "missing_token": meta.get("missing_token", MISSING_TOKEN),
        }

    meta_payload = {
        "type": "preprocessed_dataset",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_feature_dir": os.path.abspath(resolved_source) if resolved_source else "",
        "rows": int(total_rows),
        "feature_columns": feature_columns,
        "meta_columns": meta_columns_global,
        "fill_values": fill_strategies,
        "categorical_maps": categorical_serializable,
        "column_profiles": column_profiles,
        "files": [os.path.abspath(p) for p in csv_files],
    }

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_payload, fh, ensure_ascii=False, indent=2)

    _notify(progress_cb, 100)

    return {
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "meta_path": meta_path,
        "total_rows": int(total_rows),
        "total_cols": int(len(feature_columns)),
        "files": csv_files,
    }
