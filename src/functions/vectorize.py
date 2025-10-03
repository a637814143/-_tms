"""特征 CSV 的向量化工具，生成可直接用于模型训练的数据集。"""

from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import tempfile

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
    # 统一为 float32，缺失值后续在写入 numpy 数组时再处理，避免在 DataFrame
    # 阶段额外拷贝一份巨大的矩阵导致的内存峰值。
    matrix = matrix.astype("float32", copy=False)
    return matrix, categorical_maps


def _matrix_to_numpy(
    matrix: pd.DataFrame,
    *,
    dtype: str = "float32",
    allow_memmap: bool = True,
) -> Tuple[np.ndarray, Optional[str]]:
    """
    将 DataFrame 转换为 numpy 数组。

    pandas 在极宽的矩阵（列非常多，行很少）的场景下会在 ``DataFrame.to_numpy``
    中一次性申请一整块连续内存，即便数据量并不算大也可能因为内存碎片
    或平台限制导致失败。这里首先尝试直接转换；若失败则退化为手动按列
    分块复制到目标数组/内存映射文件，避免在转换过程中额外的巨大临时内存。
    """

    target_dtype = np.dtype(dtype)
    try:
        arr = matrix.to_numpy(dtype=target_dtype, copy=False)
        if np.issubdtype(arr.dtype, np.floating):
            np.nan_to_num(arr, copy=False)
        return arr, None
    except Exception as exc:  # pragma: no cover - fallback path在宽矩阵上触发
        is_memory_issue = isinstance(exc, MemoryError) or exc.__class__.__name__ == "_ArrayMemoryError"
        if not is_memory_issue and "Unable to allocate" not in str(exc):
            raise
        first_error = exc

    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        raise RuntimeError("数据为空，无法转换为数组。")

    def _iter_column_chunks(chunk_target_bytes: int):
        cells_per_chunk = max(1, chunk_target_bytes // max(1, target_dtype.itemsize))
        chunk_cols = cells_per_chunk // max(1, rows)
        if chunk_cols <= 0:
            chunk_cols = 1
        chunk_cols = min(cols, max(1, chunk_cols))

        for start in range(0, cols, chunk_cols):
            end = min(cols, start + chunk_cols)
            try:
                chunk = matrix.iloc[:, start:end].to_numpy(dtype=target_dtype, copy=False)
            except MemoryError as exc:
                raise MemoryError(
                    "向量化时内存不足，建议减少一次处理的文件数量或过滤部分特征后重试。"
                ) from exc
            if np.issubdtype(chunk.dtype, np.floating):
                np.nan_to_num(chunk, copy=False)
            yield start, end, chunk

    def _copy_into(target: np.ndarray | np.memmap) -> None:
        for start, end, chunk in _iter_column_chunks(8 * 1024 * 1024):
            try:
                target[:, start:end] = chunk
            except MemoryError as exc:
                raise MemoryError(
                    "向量化时内存不足，建议减少一次处理的文件数量或过滤部分特征后重试。"
                ) from exc

    try:
        arr = np.empty((rows, cols), dtype=target_dtype)
        _copy_into(arr)
        arr.setflags(write=False)
        return arr, None
    except MemoryError:
        if not allow_memmap:
            raise MemoryError(
                "向量化时内存不足，建议减少一次处理的文件数量或过滤部分特征后重试。"
            ) from first_error

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"vectorize_{uuid4().hex}.npy")
    try:
        mm = np.lib.format.open_memmap(
            temp_path,
            mode="w+",
            dtype=target_dtype,
            shape=(rows, cols),
        )
        _copy_into(mm)
        mm.flush()
        mm.flags.writeable = False
        return mm, temp_path
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise MemoryError(
            "向量化时内存不足，建议减少一次处理的文件数量或过滤部分特征后重试。"
        ) from first_error


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
    columns = matrix.columns.to_list()
    rows, cols = matrix.shape

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    X, memmap_path = _matrix_to_numpy(matrix)
    del matrix
    np.savez_compressed(
        output_path,
        X=X,
        columns=columns,
        source_csv=os.path.abspath(csv_path),
    )

    if isinstance(X, np.memmap):
        X.flush()
        del X
    if memmap_path:
        try:
            os.remove(memmap_path)
        except OSError:
            pass

    meta: Dict[str, object] = {
        "vector_path": output_path,
        "source_csv": os.path.abspath(csv_path),
        "rows": int(rows),
        "cols": int(cols),
        "columns": columns,
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

    max_workers = min(8, max(1, os.cpu_count() or 4))
    frames_map: Dict[str, pd.DataFrame] = {}

    if total_files > 1 and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_load_feature_csv, path): path for path in csv_files}
            for completed_idx, future in enumerate(as_completed(future_map), start=1):
                path = future_map[future]
                frames_map[path] = future.result()
                _notify(progress_cb, 5 + int(20 * completed_idx / total_files))
    else:
        for idx, csv_path in enumerate(csv_files, start=1):
            frames_map[csv_path] = _load_feature_csv(csv_path)
            _notify(progress_cb, 5 + int(20 * idx / total_files))

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
        _notify(progress_cb, 25 + int(15 * idx / total_files))

    full_df = pd.concat(frames, ignore_index=True)
    if full_df.empty:
        raise RuntimeError("聚合后的特征数据为空，无法向量化。")

    # 保留原始文件清单用于训练集切分
    metadata_cols = ["__source_file__", "__source_path__"]
    working_df = full_df.drop(columns=metadata_cols, errors="ignore")

    _notify(progress_cb, 50)
    matrix, cat_maps = _vectorize_dataframe(working_df)
    columns = matrix.columns.to_list()
    rows, cols = matrix.shape
    X, memmap_path = _matrix_to_numpy(matrix)
    del matrix
    dataset_name = f"dataset_vectors_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_path = os.path.join(output_dir, f"{dataset_name}.npz")
    np.savez_compressed(
        dataset_path,
        X=X,
        columns=columns,
    )

    if isinstance(X, np.memmap):
        X.flush()
        del X
    if memmap_path:
        try:
            os.remove(memmap_path)
        except OSError:
            pass

    manifest_path = os.path.join(output_dir, f"{dataset_name}_manifest.csv")
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")

    meta_path = os.path.join(output_dir, f"{dataset_name}_meta.json")
    meta_payload = {
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "rows": int(rows),
        "cols": int(cols),
        "columns": columns,
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
        "total_rows": int(rows),
        "total_cols": int(cols),
        "files": csv_files,
    }