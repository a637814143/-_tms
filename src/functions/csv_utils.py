"""Utility helpers for loading CSV files with multiple encodings."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


_DEFAULT_ENCODINGS: Sequence[str] = ("utf-8", "utf-8-sig", "gb18030", "latin1")


def read_csv_flexible(
    path: str | Path,
    *,
    encodings: Iterable[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a CSV file, attempting several common encodings on failure.

    Parameters
    ----------
    path:
        File path to read.
    encodings:
        Optional iterable of encodings to attempt. When omitted a default list
        covering UTF-8 (with BOM), GB18030 and Latin-1 will be used.
    **kwargs:
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        The loaded dataframe.

    Raises
    ------
    UnicodeDecodeError
        If none of the provided encodings can decode the file contents.
    Exception
        Propagates any other exception from :func:`pandas.read_csv` (e.g.
        :class:`FileNotFoundError`).
    """

    path = Path(path)
    tried_encodings = list(encodings or _DEFAULT_ENCODINGS)
    last_unicode_error: UnicodeDecodeError | None = None

    for encoding in tried_encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_unicode_error = exc
            continue

    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError as exc:
        last_unicode_error = exc

    if last_unicode_error is not None:
        detail = ", ".join(tried_encodings)
        raise UnicodeDecodeError(
            last_unicode_error.encoding or "utf-8",
            b"",
            0,
            1,
            f"无法用以下编码读取CSV: {detail}",
        ) from last_unicode_error

    # If we get here it means pandas raised a different exception without
    # providing a UnicodeDecodeError for us to surface.
    raise RuntimeError(f"无法读取CSV文件: {path}")


__all__ = ["read_csv_flexible"]
