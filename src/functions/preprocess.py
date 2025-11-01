"""Combine extracted feature CSV files into a standardized training dataset."""

from __future__ import annotations

import csv
import glob
import json
import os
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .vectorizer import (
    CSV_COLUMNS,
    CSV_READ_ENCODINGS,
    numeric_feature_names,
)

ProgressCallback = Optional[Callable[[int], None]]

FeatureSource = Union[str, Sequence[str]]


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
            resolved_source = os.path.dirname(csv_files[0]) if csv_files else None
    else:
        resolved = os.path.abspath(str(feature_dir))
        if os.path.isdir(resolved):
            patterns = ("*.csv", "*.CSV")
            for pattern in patterns:
                csv_files.extend(glob.glob(os.path.join(resolved, pattern)))
            csv_files = sorted(set(csv_files))
            resolved_source = resolved if csv_files else None
        elif os.path.isfile(resolved):
            csv_files = [resolved]
            resolved_source = os.path.dirname(resolved)
        else:
            raise FileNotFoundError(f"未找到特征数据来源: {feature_dir}")

    if not csv_files:
        raise RuntimeError("未能在所选路径中找到特征 CSV 文件。")

    return csv_files, resolved_source


def _unique_output_paths(output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"dataset_vectorized_{timestamp}"

    dataset_path = os.path.join(output_dir, f"{base_name}.csv")
    meta_path = os.path.join(output_dir, f"{base_name}_meta.json")

    counter = 1
    while os.path.exists(dataset_path) or os.path.exists(meta_path):
        base_name = f"dataset_vectorized_{timestamp}_{counter}"
        dataset_path = os.path.join(output_dir, f"{base_name}.csv")
        meta_path = os.path.join(output_dir, f"{base_name}_meta.json")
        counter += 1

    return dataset_path, meta_path


def _iter_rows(path: str) -> Iterable[List[str]]:
    last_error: Optional[UnicodeDecodeError] = None
    for encoding in CSV_READ_ENCODINGS:
        try:
            with open(path, "r", newline="", encoding=encoding) as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                if header is None:
                    return
                normalized = [column.strip() for column in header]
                if normalized != list(CSV_COLUMNS):
                    raise ValueError(f"CSV 列头不匹配: {path}")
                yield from reader
                return
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise ValueError(
        "无法读取特征 CSV，请确认编码格式是否为 UTF-8 或兼容编码。"
    ) from last_error


def preprocess_feature_dir(
    feature_dir: FeatureSource,
    output_dir: str,
    *,
    progress_cb: ProgressCallback = None,
) -> Dict[str, object]:
    """Merge feature CSV files into a single, schema-verified dataset."""

    csv_files, resolved_source = _resolve_feature_sources(feature_dir)
    dataset_path, meta_path = _unique_output_paths(output_dir)

    total_files = len(csv_files)
    total_rows = 0

    with open(dataset_path, "w", newline="", encoding="utf-8") as out_handle:
        writer = csv.writer(out_handle)
        writer.writerow(CSV_COLUMNS)

        for index, path in enumerate(csv_files, start=1):
            for row in _iter_rows(path):
                if not row or all(cell.strip() == "" for cell in row):
                    continue
                if len(row) != len(CSV_COLUMNS):
                    raise ValueError(
                        f"CSV 列数不匹配: {path} (expected {len(CSV_COLUMNS)})"
                    )
                writer.writerow(row)
                total_rows += 1

            _notify(progress_cb, int(index / total_files * 95))

    _notify(progress_cb, 100)

    feature_columns = numeric_feature_names()

    meta_payload: Dict[str, object] = {
        "schema_version": "2025.10",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_feature_dir": os.path.abspath(resolved_source) if resolved_source else "",
        "source_files": [os.path.abspath(path) for path in csv_files],
        "total_rows": total_rows,
        "total_columns": len(CSV_COLUMNS),
        "feature_columns": feature_columns,
        "csv_columns": list(CSV_COLUMNS),
    }

    with open(meta_path, "w", encoding="utf-8") as meta_handle:
        json.dump(meta_payload, meta_handle, ensure_ascii=False, indent=2)

    return {
        "dataset_path": dataset_path,
        "manifest_path": dataset_path,
        "meta_path": meta_path,
        "total_rows": total_rows,
        "total_cols": len(feature_columns),
        "feature_columns": feature_columns,
        "files": csv_files,
    }

