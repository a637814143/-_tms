"""特征 CSV 数据预处理，将其整理成标准化的训练数据集。"""

from __future__ import annotations

import csv
import glob
import json
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

try:  # pandas 在运行环境中是可选的
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas 可能未安装
    pd = None  # type: ignore

from .csv_utils import read_csv_flexible
from .vectorizer import CSV_COLUMNS

ProgressCallback = Optional[Callable[[int], None]]
FeatureSource = Union[str, Sequence[str]]

_STRING_COLUMNS = {"Flow ID", "Source IP", "Destination IP", "Timestamp"}
_LABEL_COLUMN = "Label"
_DATASET_META_TYPE = "merged_feature_dataset"


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


def preprocess_feature_dir(
    feature_dir: FeatureSource,
    output_dir: str,
    *,
    progress_cb: ProgressCallback = None,
) -> Dict[str, object]:
    """批量读取特征 CSV，合并为单一且列顺序固定的数据集。"""

    if pd is None:  # pragma: no cover - 仅在缺少 pandas 时触发
        raise RuntimeError("pandas 未安装，无法执行数据预处理。")

    csv_files, resolved_source = _resolve_feature_sources(feature_dir)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"dataset_{timestamp}"
    dataset_path = os.path.join(output_dir, f"{base_name}.csv")
    manifest_path = os.path.join(output_dir, f"{base_name}_manifest.csv")
    meta_path = os.path.join(output_dir, f"{base_name}_meta.json")

    counter = 1
    while os.path.exists(dataset_path):
        base_name = f"dataset_{timestamp}_{counter}"
        dataset_path = os.path.join(output_dir, f"{base_name}.csv")
        manifest_path = os.path.join(output_dir, f"{base_name}_manifest.csv")
        meta_path = os.path.join(output_dir, f"{base_name}_meta.json")
        counter += 1

    header = list(CSV_COLUMNS)
    with open(dataset_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

    manifest_rows: List[Dict[str, object]] = []
    total_rows = 0
    labeled_rows = 0
    total_files = len(csv_files)

    for index, csv_path in enumerate(csv_files, start=1):
        df = read_csv_flexible(csv_path)
        aligned = _align_dataframe(df)
        aligned.to_csv(dataset_path, mode="a", header=False, index=False, encoding="utf-8")

        rows = int(aligned.shape[0])
        if rows:
            label_series = aligned[_LABEL_COLUMN].astype(str)
            labeled = int(label_series.str.strip().ne("").sum())
        else:
            labeled = 0

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

    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = ["source_file", "source_path", "rows", "labeled_rows"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

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