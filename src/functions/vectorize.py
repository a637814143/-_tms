"""特征 CSV 的向量化工具，生成可直接用于模型训练的数据集。"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

ProgressCallback = Optional[Callable[[int], None]]


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


def _vectorize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    """将任意 DataFrame 转换为仅包含数值列的矩阵，并返回分类列映射。"""

    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    numeric_part = pd.DataFrame(index=df.index)
    if numeric_cols:
        numeric_part = df[numeric_cols].apply(pd.to_numeric, errors="coerce").astype("float32")

    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    categorical_maps: Dict[str, List[str]] = {}
    categorical_parts = []

    for col in categorical_cols:
        series = df[col]
        # 将缺失值统一标记，保证 factorize 不返回 -1
        filled = series.fillna("<NA>")
        if filled.dtype == object:
            normalized = filled.astype(str)
        else:
            normalized = filled.astype(str)
        codes, uniques = pd.factorize(normalized, sort=True)
        categorical_maps[col] = [str(u) for u in uniques]
        categorical_parts.append(pd.Series(codes.astype("float32"), name=f"{col}__id"))

    pieces = []
    if not numeric_part.empty:
        pieces.append(numeric_part)
    if categorical_parts:
        pieces.append(pd.concat(categorical_parts, axis=1))

    if not pieces:
        raise ValueError("未找到可向量化的列")

    matrix = pd.concat(pieces, axis=1)
    matrix = matrix.fillna(0.0).astype("float32")
    return matrix, categorical_maps


def vectorize_csv(
    csv_path: str,
    output_path: str,
    *,
    progress_cb: ProgressCallback = None,
) -> Dict[str, object]:
    """将单个特征 CSV 向量化并保存为 .npz。"""

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    _notify(progress_cb, 5)
    df = _load_feature_csv(csv_path)

    # 不参与训练但需要在清单中保留的列
    meta_cols = ["__source_file__", "__source_path__"]
    clean_df = df.drop(columns=meta_cols, errors="ignore")

    _notify(progress_cb, 35)
    matrix, cat_maps = _vectorize_dataframe(clean_df)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    X = matrix.to_numpy(dtype="float32", copy=False)
    np.savez_compressed(
        output_path,
        X=X,
        columns=matrix.columns.to_list(),
        source_csv=os.path.abspath(csv_path),
    )

    meta: Dict[str, object] = {
        "vector_path": output_path,
        "source_csv": os.path.abspath(csv_path),
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "columns": matrix.columns.to_list(),
    }

    if cat_maps:
        cat_path = output_path.replace(".npz", "_cats.json")
        with open(cat_path, "w", encoding="utf-8") as fh:
            json.dump(cat_maps, fh, ensure_ascii=False, indent=2)
        meta["categorical_map_path"] = cat_path

    _notify(progress_cb, 100)
    return meta


def vectorize_feature_dir(
    feature_dir: str,
    output_dir: str,
    *,
    progress_cb: ProgressCallback = None,
) -> Dict[str, object]:
    """批量读取特征 CSV 并生成统一的训练数据集。"""

    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"目录不存在: {feature_dir}")

    patterns = ["*.csv", "*.CSV"]
    csv_files = []
    for pattern in patterns:
        csv_files.extend(glob.glob(os.path.join(feature_dir, pattern)))
    csv_files = sorted(set(csv_files))

    if not csv_files:
        raise RuntimeError(f"目录下没有找到特征 CSV: {feature_dir}")

    os.makedirs(output_dir, exist_ok=True)

    frames: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, object]] = []
    total_files = len(csv_files)
    row_cursor = 0

    for idx, csv_path in enumerate(csv_files, start=1):
        df = _load_feature_csv(csv_path)
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
        _notify(progress_cb, 10 + int(30 * idx / total_files))

    full_df = pd.concat(frames, ignore_index=True)
    if full_df.empty:
        raise RuntimeError("聚合后的特征数据为空，无法向量化。")

    # 保留原始文件清单用于训练集切分
    metadata_cols = ["__source_file__", "__source_path__"]
    working_df = full_df.drop(columns=metadata_cols, errors="ignore")

    _notify(progress_cb, 50)
    matrix, cat_maps = _vectorize_dataframe(working_df)

    X = matrix.to_numpy(dtype="float32", copy=False)
    dataset_name = f"dataset_vectors_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_path = os.path.join(output_dir, f"{dataset_name}.npz")
    np.savez_compressed(
        dataset_path,
        X=X,
        columns=matrix.columns.to_list(),
    )

    manifest_path = os.path.join(output_dir, f"{dataset_name}_manifest.csv")
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")

    meta_path = os.path.join(output_dir, f"{dataset_name}_meta.json")
    meta_payload = {
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
        "columns": matrix.columns.to_list(),
    }
    if cat_maps:
        cat_map_path = os.path.join(output_dir, f"{dataset_name}_cats.json")
        with open(cat_map_path, "w", encoding="utf-8") as fh:
            json.dump(cat_maps, fh, ensure_ascii=False, indent=2)
        meta_payload["categorical_map_path"] = cat_map_path

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_payload, fh, ensure_ascii=False, indent=2)

    _notify(progress_cb, 100)

    return {
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "meta_path": meta_path,
        "total_rows": int(X.shape[0]),
        "total_cols": int(X.shape[1]),
        "files": csv_files,
    }
