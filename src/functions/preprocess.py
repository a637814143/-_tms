"""特征 CSV 数据预处理，将其整理成可直接用于模型训练的标准化数据集。"""

from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Dict, List, Optional

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


def _prepare_numeric(series: pd.Series) -> tuple[pd.Series, Dict[str, float]]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        fill_value = 0.0
    else:
        fill_value = float(numeric.median(skipna=True))
        if np.isnan(fill_value):
            fill_value = float(numeric.dropna().iloc[0]) if numeric.dropna().any() else 0.0
    filled = numeric.fillna(fill_value).astype("float32")
    stats = {
        "fill_value": float(fill_value),
        "min": float(filled.min(initial=0.0)),
        "max": float(filled.max(initial=0.0)),
        "mean": float(filled.mean()),
        "std": float(filled.std(ddof=0)),
    }
    return filled, stats


def _prepare_boolean(series: pd.Series) -> tuple[pd.Series, Dict[str, float]]:
    bool_series = series.fillna(False).astype(bool)
    numeric = bool_series.astype("int32")
    stats = {
        "fill_value": 0.0,
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
    }
    return numeric, stats


def _prepare_categorical(series: pd.Series, column: str) -> tuple[pd.Series, Dict[str, object]]:
    filled = series.fillna(MISSING_TOKEN)
    if is_datetime64_any_dtype(filled):
        filled = filled.astype(str)
    elif not is_categorical_dtype(filled):
        filled = filled.astype(str)
    normalized = filled.str.strip().replace("", MISSING_TOKEN)
    codes, uniques = pd.factorize(normalized, sort=True)
    encoded = pd.Series(codes.astype("int32"), name=f"{column}__code", index=series.index)
    metadata = {
        "source_column": column,
        "labels": [str(u) for u in uniques],
        "missing_token": MISSING_TOKEN,
    }
    return encoded, metadata


def preprocess_feature_dir(
    feature_dir: str,
    output_dir: str,
    *,
    progress_cb: ProgressCallback = None,
) -> Dict[str, object]:
    """批量读取特征 CSV，执行数据预处理并输出统一的训练数据集。"""

    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"目录不存在: {feature_dir}")

    patterns = ["*.csv", "*.CSV"]
    csv_files: List[str] = []
    for pattern in patterns:
        csv_files.extend(glob.glob(os.path.join(feature_dir, pattern)))
    csv_files = sorted(set(csv_files))

    if not csv_files:
        raise RuntimeError(f"目录下没有找到特征 CSV: {feature_dir}")

    os.makedirs(output_dir, exist_ok=True)

    frames_map: Dict[str, pd.DataFrame] = {}
    total_files = len(csv_files)
    max_workers = min(8, max(1, os.cpu_count() or 4))

    if total_files > 1 and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_load_feature_csv, path): path for path in csv_files}
            for idx, future in enumerate(as_completed(future_map), start=1):
                path = future_map[future]
                frames_map[path] = future.result()
                _notify(progress_cb, 5 + int(20 * idx / total_files))
    else:
        for idx, csv_path in enumerate(csv_files, start=1):
            frames_map[csv_path] = _load_feature_csv(csv_path)
            _notify(progress_cb, 5 + int(20 * idx / total_files))

    frames: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, object]] = []
    row_cursor = 0

    for idx, csv_path in enumerate(csv_files, start=1):
        df = frames_map[csv_path]
        df["__source_file__"] = os.path.basename(csv_path)
        df["__source_path__"] = os.path.abspath(csv_path)
        frames.append(df)

        rows = len(df)
        manifest_rows.append(
            {
                "source_file": os.path.basename(csv_path),
                "source_path": os.path.abspath(csv_path),
                "start_index": row_cursor,
                "end_index": row_cursor + rows,
                "rows": rows,
            }
        )
        row_cursor += rows
        _notify(progress_cb, 30 + int(15 * idx / total_files))

    full_df = pd.concat(frames, ignore_index=True)
    if full_df.empty:
        raise RuntimeError("聚合后的特征数据为空，无法进行预处理。")

    meta_cols_in_df = [col for col in RESERVED_META_COLUMNS if col in full_df.columns]

    working_df = full_df.copy()

    processed_columns: Dict[str, pd.Series] = {}
    column_profiles: List[Dict[str, object]] = []
    categorical_maps: Dict[str, Dict[str, object]] = {}
    fill_strategies: Dict[str, float] = {}

    for column in working_df.columns:
        if column in meta_cols_in_df:
            continue

        series = working_df[column]

        if is_bool_dtype(series):
            encoded, stats = _prepare_boolean(series)
            processed_columns[column] = encoded
            column_profiles.append(
                {
                    "name": column,
                    "kind": "boolean",
                    **stats,
                }
            )
            fill_strategies[column] = stats["fill_value"]
        elif is_numeric_dtype(series):
            encoded, stats = _prepare_numeric(series)
            processed_columns[column] = encoded
            column_profiles.append(
                {
                    "name": column,
                    "kind": "numeric",
                    **stats,
                }
            )
            fill_strategies[column] = stats["fill_value"]
        else:
            encoded, meta = _prepare_categorical(series, column)
            processed_columns[encoded.name] = encoded
            categorical_maps[encoded.name] = meta
            column_profiles.append(
                {
                    "name": encoded.name,
                    "kind": "categorical_encoded",
                    "source_column": column,
                    "unique": len(meta["labels"]),
                }
            )

    if not processed_columns:
        raise RuntimeError("没有找到可用于训练的特征列。")

    feature_df = pd.DataFrame(processed_columns, index=working_df.index)

    if meta_cols_in_df:
        meta_df = full_df[meta_cols_in_df]
        processed_df = pd.concat([meta_df, feature_df], axis=1)
    else:
        processed_df = feature_df

    dataset_name = f"dataset_preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_path = os.path.join(output_dir, f"{dataset_name}.csv")
    manifest_path = os.path.join(output_dir, f"{dataset_name}_manifest.csv")
    meta_path = os.path.join(output_dir, f"{dataset_name}_meta.json")

    processed_df.to_csv(dataset_path, index=False, encoding="utf-8")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")

    feature_columns = [col for col in processed_df.columns if col not in meta_cols_in_df]

    meta_payload = {
        "type": "preprocessed_dataset",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_feature_dir": os.path.abspath(feature_dir),
        "rows": int(len(processed_df)),
        "feature_columns": feature_columns,
        "meta_columns": meta_cols_in_df,
        "fill_values": fill_strategies,
        "categorical_maps": categorical_maps,
        "column_profiles": column_profiles,
    }

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_payload, fh, ensure_ascii=False, indent=2)

    _notify(progress_cb, 100)

    return {
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "meta_path": meta_path,
        "total_rows": int(len(processed_df)),
        "total_cols": int(len(feature_columns)),
        "files": csv_files,
    }
