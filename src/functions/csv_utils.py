"""CSV helpers used by the legacy GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

_FALLBACK_ENCODINGS = [
    "utf-8",
    "utf-8-sig",
    "gbk",
    "gb2312",
    "latin-1",
]


def read_csv_flexible(path: str, *, nrows: Optional[int] = None) -> pd.DataFrame:
    file_path = Path(path)
    last_error: Optional[Exception] = None
    for encoding in _FALLBACK_ENCODINGS:
        try:
            return pd.read_csv(file_path, nrows=nrows, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    # Fallback: let pandas try with replacement to avoid crashing the GUI
    return pd.read_csv(file_path, nrows=nrows, encoding="utf-8", encoding_errors="replace")
