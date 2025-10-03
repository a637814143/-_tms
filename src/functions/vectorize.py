"""特征 CSV 的向量化工具，生成可直接用于模型训练的数据集。"""

from __future__ import annotations

import glob
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

ProgressCallback = Optional[Callable[[int], None]]


def _notify(cb: ProgressCallback, value: int) -> None:
    if cb:
        cb(max(0, min(100, int(value))))


def _is_auxiliary_csv(path: str) -> bool:
    name = os.path.basename(path).lower()
    return any(
        name.endswith(suffix)
        for suffix in (
            "_manifest.csv",
            "_meta.csv",
        )
    )


def _iter_csv_chunks(csv_path: str, *, chunk_size: int) -> Iterable[pd.DataFrame]:
    try:
        reader = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            encoding="utf-8",
            low_memory=False,
        )
    except UnicodeDecodeError:
        reader = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            encoding="gbk",
            low_memory=False,
        )
    for chunk in reader:
        yield chunk


def _normalize_str_array(series: pd.Series) -> np.ndarray:
    filled = series.fillna("<NA>")
    return filled.astype(str).to_numpy(dtype=object, copy=False)


def _series_is_numeric(series: pd.Series) -> bool:
    if series.empty:
        return True
    if is_numeric_dtype(series) or is_bool_dtype(series):
        return True
    if series.dtype == object:
        converted = pd.to_numeric(series, errors="coerce")
        non_na = int(converted.notna().sum())
        if non_na == 0:
            return False
        return non_na >= max(1, int(0.9 * len(series)))
    return False


def _suggest_chunk_rows(csv_files: List[str], metadata_cols: List[str]) -> int:
    target_bytes = 32 * 1024 * 1024  # 32MB 估计值
    for path in csv_files:
        try:
            sample = pd.read_csv(path, nrows=1, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            sample = pd.read_csv(path, nrows=1, encoding="gbk", low_memory=False)
        sample = sample.drop(columns=metadata_cols, errors="ignore")
        cols = len(sample.columns)
        if cols:
            approx_row_bytes = max(1, cols) * 8  # 预估 float64 大小
            rows = max(1, target_bytes // approx_row_bytes)
            return int(max(1, min(rows, 20000)))
    return 1000


@dataclass
class VectorizationPlan:
    csv_files: List[str]
    metadata_cols: List[str]
    numeric_order: List[str] = field(default_factory=list)
    categorical_order: List[str] = field(default_factory=list)
    categorical_maps: Dict[str, Dict[str, int]] = field(default_factory=dict)
    total_rows: int = 0

    @property
    def columns(self) -> List[str]:
        return self.numeric_order + [f"{col}__id" for col in self.categorical_order]

    @property
    def total_cols(self) -> int:
        return len(self.columns)


def _plan_vectorization(
    csv_files: List[str],
    *,
    metadata_cols: List[str],
    chunk_size: int,
    progress_cb: ProgressCallback = None,
) -> VectorizationPlan:
    if not csv_files:
        raise RuntimeError("没有可供向量化的特征 CSV 文件")

    plan = VectorizationPlan(csv_files=list(csv_files), metadata_cols=metadata_cols)
    total_files = len(csv_files)

    for file_idx, path in enumerate(csv_files, start=1):
        rows_this_file = 0
        for chunk in _iter_csv_chunks(path, chunk_size=chunk_size):
            chunk = chunk.drop(columns=metadata_cols, errors="ignore")
            if chunk.empty:
                continue

            rows = len(chunk)
            rows_this_file += rows
            plan.total_rows += rows

            for col in chunk.columns:
                series = chunk[col]
                if col not in plan.numeric_order and col not in plan.categorical_order:
                    if _series_is_numeric(series):
                        plan.numeric_order.append(col)
                    else:
                        plan.categorical_order.append(col)
                        plan.categorical_maps[col] = {}
                if col in plan.categorical_maps:
                    mapping = plan.categorical_maps[col]
                    for value in pd.unique(_normalize_str_array(series)):
                        val_str = str(value)
                        if val_str not in mapping:
                            mapping[val_str] = len(mapping)

        if rows_this_file == 0:
            # 即使空文件也进行进度推进，避免卡在 0%
            pass
        if progress_cb:
            fraction = file_idx / total_files
            _notify(progress_cb, 5 + int(35 * fraction))

    if plan.total_rows == 0:
        raise RuntimeError("特征 CSV 中没有任何可用数据，无法向量化")

    for mapping in plan.categorical_maps.values():
        if "<NA>" not in mapping:
            mapping["<NA>"] = len(mapping)

    if plan.total_cols == 0:
        raise RuntimeError("未找到任何可向量化的数值或分类列")

    return plan


def _ordered_categorical_lists(raw_maps: Dict[str, Dict[str, int]]) -> Dict[str, List[str]]:
    ordered: Dict[str, List[str]] = {}
    for col, mapping in raw_maps.items():
        values = [None] * len(mapping)
        for value, idx in mapping.items():
            values[idx] = str(value)
        ordered[col] = values
    return ordered


def _write_vector_dataset(
    plan: VectorizationPlan,
    dataset_path: str,
    *,
    chunk_size: int,
    create_manifest: bool,
    progress_cb: ProgressCallback = None,
) -> Optional[List[Dict[str, object]]]:
    os.makedirs(os.path.dirname(dataset_path) or ".", exist_ok=True)

    rows = plan.total_rows
    cols = plan.total_cols
    bytes_per_value = 4  # float32
    target_chunk_bytes = 128 * 1024 * 1024  # 128MB
    chunk_rows = max(1, int(target_chunk_bytes / max(1, cols) / bytes_per_value))
    chunk_rows = max(1, min(chunk_rows, chunk_size))

    temp_fd, temp_path = tempfile.mkstemp(prefix="vector_tmp_", suffix=".npy", dir=os.path.dirname(dataset_path) or None)
    os.close(temp_fd)

    manifest_rows: Optional[List[Dict[str, object]]] = [] if create_manifest else None
    row_cursor = 0

    try:
        mmap = np.lib.format.open_memmap(temp_path, mode="w+", dtype="float32", shape=(rows, cols))

        for file_idx, path in enumerate(plan.csv_files, start=1):
            file_start = row_cursor
            for chunk in _iter_csv_chunks(path, chunk_size=chunk_rows):
                chunk = chunk.drop(columns=plan.metadata_cols, errors="ignore")
                if chunk.empty:
                    continue

                batch_rows = len(chunk)
                numeric_data = np.zeros((batch_rows, len(plan.numeric_order)), dtype=np.float32)
                for col_idx, col in enumerate(plan.numeric_order):
                    if col in chunk.columns:
                        numeric_series = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0)
                        numeric_data[:, col_idx] = numeric_series.to_numpy(dtype=np.float32, copy=False)
                    else:
                        numeric_data[:, col_idx] = 0.0

                if plan.categorical_order:
                    cat_data = np.zeros((batch_rows, len(plan.categorical_order)), dtype=np.float32)
                    for cat_idx, col in enumerate(plan.categorical_order):
                        mapping = plan.categorical_maps[col]
                        if col in chunk.columns:
                            values = _normalize_str_array(chunk[col])
                        else:
                            values = np.full(batch_rows, "<NA>", dtype=object)
                        ids = np.fromiter(
                            (mapping.get(str(val), mapping["<NA>"]) for val in values),
                            dtype=np.int32,
                            count=batch_rows,
                        )
                        cat_data[:, cat_idx] = ids.astype(np.float32, copy=False)
                    batch_matrix = np.hstack([numeric_data, cat_data])
                else:
                    batch_matrix = numeric_data

                mmap[row_cursor : row_cursor + batch_rows, :] = batch_matrix
                row_cursor += batch_rows

                if progress_cb and plan.total_rows:
                    processed_fraction = row_cursor / plan.total_rows
                    _notify(progress_cb, 40 + int(55 * processed_fraction))

            if manifest_rows is not None:
                file_rows = row_cursor - file_start
                manifest_rows.append(
                    {
                        "source_file": os.path.basename(path),
                        "source_path": os.path.abspath(path),
                        "start_index": file_start,
                        "end_index": (row_cursor - 1) if file_rows > 0 else (file_start - 1),
                        "rows": file_rows,
                    }
                )

        del mmap

        mmap = np.lib.format.open_memmap(temp_path, mode="r", dtype="float32", shape=(rows, cols))
        np.savez_compressed(dataset_path, X=mmap, columns=plan.columns)
        del mmap
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    if progress_cb:
        _notify(progress_cb, 96)

    return manifest_rows
 
 
def vectorize_csv(
    csv_path: str,
    output_path: str,
    *,
    progress_cb: ProgressCallback = None,
) -> Dict[str, object]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    metadata_cols = ["__source_file__", "__source_path__"]
    chunk_rows = _suggest_chunk_rows([csv_path], metadata_cols)
    plan = _plan_vectorization(
        [csv_path],
        metadata_cols=metadata_cols,
        chunk_size=chunk_rows,
        progress_cb=progress_cb,
    )

    _ = _write_vector_dataset(
        plan,
        output_path,
        chunk_size=chunk_rows,
        create_manifest=False,
        progress_cb=progress_cb,
    )

    cat_lists = _ordered_categorical_lists(plan.categorical_maps)

    meta: Dict[str, object] = {
        "vector_path": output_path,
        "source_csv": os.path.abspath(csv_path),
        "rows": int(plan.total_rows),
        "cols": int(plan.total_cols),
        "columns": plan.columns,
    }

    if cat_lists:
        cat_path = output_path.replace(".npz", "_cats.json")
        with open(cat_path, "w", encoding="utf-8") as fh:
            json.dump(cat_lists, fh, ensure_ascii=False, indent=2)
        meta["categorical_map_path"] = cat_path

    if progress_cb:
        _notify(progress_cb, 100)

    return meta

def vectorize_feature_dir(
    feature_dir: str,
    output_dir: str,
    *,
    progress_cb: ProgressCallback = None,
) -> Dict[str, object]:
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"目录不存在: {feature_dir}")

    patterns = ["*.csv", "*.CSV"]
    csv_files = []
    for pattern in patterns:
        csv_files.extend(glob.glob(os.path.join(feature_dir, pattern)))
    csv_files = sorted({path for path in csv_files if not _is_auxiliary_csv(path)})

    if not csv_files:
        raise RuntimeError(f"目录下没有找到特征 CSV: {feature_dir}")

    os.makedirs(output_dir, exist_ok=True)

    metadata_cols = ["__source_file__", "__source_path__"]
    chunk_rows = _suggest_chunk_rows(csv_files, metadata_cols)

    plan = _plan_vectorization(
        csv_files,
        metadata_cols=metadata_cols,
        chunk_size=chunk_rows,
        progress_cb=progress_cb,
    )

    dataset_name = f"dataset_vectors_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_path = os.path.join(output_dir, f"{dataset_name}.npz")

    manifest_rows = _write_vector_dataset(
        plan,
        dataset_path,
        chunk_size=chunk_rows,
        create_manifest=True,
        progress_cb=progress_cb,
    )

    cat_lists = _ordered_categorical_lists(plan.categorical_maps)
    cat_map_path = None
    if cat_lists:
        cat_map_path = os.path.join(output_dir, f"{dataset_name}_cats.json")
        with open(cat_map_path, "w", encoding="utf-8") as fh:
            json.dump(cat_lists, fh, ensure_ascii=False, indent=2)

    manifest_path = os.path.join(output_dir, f"{dataset_name}_manifest.csv")
    if manifest_rows is None:
        manifest_rows = []
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8")

    meta_path = os.path.join(output_dir, f"{dataset_name}_meta.json")
    meta_payload = {
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "rows": int(plan.total_rows),
        "cols": int(plan.total_cols),
        "columns": plan.columns,
    }
    if cat_map_path:
        meta_payload["categorical_map_path"] = cat_map_path
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_payload, fh, ensure_ascii=False, indent=2)

    if progress_cb:
        _notify(progress_cb, 100)

    result: Dict[str, object] = {
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "meta_path": meta_path,
        "total_rows": int(plan.total_rows),
        "total_cols": int(plan.total_cols),
        "files": csv_files,
    }
    if cat_map_path:
        result["categorical_map_path"] = cat_map_path

    return result
