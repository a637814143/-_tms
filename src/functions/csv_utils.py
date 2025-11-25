"""Utility helpers for loading CSV files with multiple encodings."""

from __future__ import annotations

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


__all__ = ["read_csv_flexible"]