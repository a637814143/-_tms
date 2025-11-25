"""Utility helpers for loading CSV files with multiple encodings."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, cast

try:  # pandas is an optional dependency at runtime
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover - pandas may be absent in minimal envs
    pd = None  # type: ignore
    _PANDAS_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when pandas is available
    _PANDAS_IMPORT_ERROR = None

_DEFAULT_ENCODINGS: Sequence[str] = ("utf-8", "utf-8-sig", "gb18030", "latin1")


def _normalise_encodings(encodings: Iterable[Optional[str]]) -> list[Optional[str]]:
    seen: set[Optional[str]] = set()
    ordered: list[Optional[str]] = []
    for encoding in encodings:
        if encoding in seen:
            continue
        ordered.append(encoding)
        seen.add(encoding)
    return ordered


def read_csv_flexible(
    path: Union[str, Path],
    *,
    encodings: Optional[Iterable[str]] = None,
    columns: Optional[Sequence[str]] = None,
    **kwargs,
) -> "pd.DataFrame":
    """Load a CSV file, attempting several common encodings on failure.

    Parameters
    ----------
    path:
        File path to read.
    encodings:
        Optional iterable of encodings to attempt. When omitted a default list
        covering UTF-8 (with BOM), GB18030 and Latin-1 will be used.
    columns:
        Optional column ordering constraint applied after the CSV has been read.
    **kwargs:
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        The loaded dataframe.

    Raises
    ------
    RuntimeError
        If pandas is unavailable in the current environment.
    UnicodeDecodeError
        If none of the provided encodings can decode the file contents.
    KeyError
        When ``columns`` is provided but missing from the loaded dataframe.
    """

    if pd is None:  # pragma: no cover - triggered only when pandas is missing
        raise RuntimeError("pandas 未安装，无法读取 CSV 文件。") from _PANDAS_IMPORT_ERROR

    csv_path = Path(path)
    base_kwargs = dict(kwargs)
    column_order = list(columns) if columns is not None else None

    preferred_encoding = base_kwargs.pop("encoding", None)
    encoding_candidates: list[Optional[str]] = []
    if preferred_encoding:
        encoding_candidates.append(cast(str, preferred_encoding))
    if encodings is not None:
        encoding_candidates.extend(encodings)
    else:
        encoding_candidates.extend(_DEFAULT_ENCODINGS)
    encoding_candidates = _normalise_encodings(encoding_candidates)
    if not encoding_candidates:
        encoding_candidates = [None]
    elif None not in encoding_candidates:
        encoding_candidates.append(None)

    last_unicode_error: Optional[UnicodeDecodeError] = None
    tried_encodings: list[str] = []
    for encoding in encoding_candidates:
        attempt_kwargs = dict(base_kwargs)
        if encoding is not None:
            attempt_kwargs["encoding"] = encoding
        try:
            df = pd.read_csv(csv_path, index_col=False, **attempt_kwargs)
            df.columns = [str(col).strip() for col in df.columns]
        except UnicodeDecodeError as exc:
            tried_encodings.append(encoding or "<default>")
            last_unicode_error = exc
            continue
        except ValueError as exc:
            # ``usecols`` errors are more helpful when columns were provided.
            if column_order is not None and "Usecols" in str(exc):
                missing = [col for col in column_order if col not in getattr(exc, "columns", [])]
                if missing:
                    raise KeyError(f"CSV 缺少列: {', '.join(missing)}") from exc
            raise
        else:
            if encoding is not None:
                df.attrs["encoding"] = encoding
            if column_order is not None:
                missing = [col for col in column_order if col not in df.columns]
                if missing:
                    raise KeyError(f"CSV 缺少列: {', '.join(missing)}")
                df = df.loc[:, column_order]
            return df

    if last_unicode_error is not None:
        detail = ", ".join(tried_encodings) or "未知编码"
        raise UnicodeDecodeError(
            last_unicode_error.encoding or "utf-8",
            last_unicode_error.object,
            last_unicode_error.start,
            last_unicode_error.end,
            f"无法使用提供的编码读取文件（尝试：{detail}）",
        ) from last_unicode_error

    # Fallback safeguard: propagate the most recent exception if it exists.
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "无法解码 CSV 文件。")


def fix_dataset_label_columns(csv_path: Union[str, Path]) -> Path:
    """
    修复类似 dataset_20251125_202623.csv 这种情况：
    - 表头有 86 列（最后是 Label, LabelBinary）
    - 每一行有 88 个值
    - 多出来的两列在第 77、78 列（下标 76、77），
      导致 Active/Idle/Label 整体错位。

    修复逻辑：
    - 保留前 76 列不动（下标 0..75）
    - 丢弃下标 76、77 这两列多余数据
    - 把后面的 10 个值（下标 78..87）拼回去
      => 对齐到表头的 Active Mean ~ LabelBinary
    - 写出 *_fixed.csv，返回新文件路径。
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 CSV 文件: {path}")

    output = path.with_name(path.stem + "_fixed" + path.suffix)

    with path.open("r", encoding="utf-8", newline="") as fin, \
         output.open("w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV 文件为空: {path}") from None

        num_cols = len(header)
        if num_cols != 86:
            raise ValueError(
                f"当前修复逻辑假定表头有 86 列，实际为 {num_cols} 列，请确认文件是否为预期的数据集。"
            )

        # 这里我们保持表头不变：Flow ID ... Label, LabelBinary
        writer.writerow(header)

        line_no = 1  # 已读过 header
        for row in reader:
            line_no += 1
            if not row:
                continue

            if len(row) != 88:
                raise ValueError(
                    f"第 {line_no} 行列数为 {len(row)}，而不是预期的 88，"
                    f"说明文件结构不符合当前修复假设。"
                )

            # 前 76 列（0..75）是正常特征
            features_front = row[:76]
            # 中间的 row[76:78] 是多出来的两列，直接丢弃
            # 后面的 10 个值 row[78:]：依次对应 Active Mean ~ LabelBinary
            tail = row[78:]

            new_row = features_front + tail  # 长度 = 76 + 10 = 86
            writer.writerow(new_row)

    return output


__all__ = ["read_csv_flexible", "fix_dataset_label_columns"]