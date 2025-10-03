"""特征 CSV 数据预处理，将其整理成可直接用于模型训练的标准化数据集。"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime

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


            numeric = pd.to_numeric(series, errors="coerce")
        else:
            filled = series.fillna(MISSING_TOKEN)
            if is_datetime64_any_dtype(filled):
                filled = filled.astype(str)
            elif not is_categorical_dtype(filled):
                filled = filled.astype(str)
            normalized = filled.str.strip().replace("", MISSING_TOKEN)
                    "source_column": column,
                    "missing_token": MISSING_TOKEN,


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

    total_files = len(csv_files)

    for idx, csv_path in enumerate(csv_files, start=1):
        df["__source_file__"] = os.path.basename(csv_path)
        df["__source_path__"] = os.path.abspath(csv_path)

        manifest_rows.append(
            {
                "source_file": os.path.basename(csv_path),
                "source_path": os.path.abspath(csv_path),
                "rows": rows,
            }
        )




    column_profiles: List[Dict[str, object]] = []
            column_profiles.append(
                {
                    "name": column,
                    "kind": "numeric",
                }
            )
        else:
            column_profiles.append(
                {
                    "kind": "categorical_encoded",
                }
            )


    meta_payload = {
        "type": "preprocessed_dataset",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_feature_dir": os.path.abspath(feature_dir),
        "feature_columns": feature_columns,
        "fill_values": fill_strategies,
        "column_profiles": column_profiles,
    }

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_payload, fh, ensure_ascii=False, indent=2)

    _notify(progress_cb, 100)

    return {
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "meta_path": meta_path,
        "total_cols": int(len(feature_columns)),
        "files": csv_files,
    }