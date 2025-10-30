"""PCAP feature preview helpers used by the legacy GUI."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import pandas as pd

from PCAP import extract_sources_to_jsonl, vectorize_jsonl_files
from src.configuration import project_root

ProgressCallback = Optional[Callable[[int], None]]
CancelCallback = Optional[Callable[[], bool]]


def _artifacts_dir() -> Path:
    base = project_root() / "artifacts" / "pcap_preview"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _iter_sources(path: str, files: Optional[Sequence[str]]) -> List[str]:
    if files:
        return [str(Path(p)) for p in files]
    path_obj = Path(path)
    if path_obj.is_dir():
        return [str(p) for p in path_obj.rglob("*.pcap*") if p.is_file()]
    return [str(path_obj)]


def get_pcap_features(
    path: str,
    *,
    files: Optional[Sequence[str]] = None,
    workers: Optional[int] = None,
    progress_cb: ProgressCallback = None,
    cancel_cb: CancelCallback = None,
    **_: object,
):
    """Extract PCAP flow previews into a DataFrame for the GUI table."""

    sources = _iter_sources(path, files)
    frames: List[pd.DataFrame] = []
    errors: List[str] = []

    total = max(1, len(sources))
    for idx, src in enumerate(sources, 1):
        if cancel_cb and cancel_cb():
            break
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                jsonl_path = Path(tmpdir) / "features.jsonl"
                extract_sources_to_jsonl(
                    src,
                    jsonl_path,
                    max_workers=workers,
                    progress_callback=None,
                )
                csv_path = Path(tmpdir) / "features.csv"
                vectorize_jsonl_files([jsonl_path], csv_path, show_progress=False)
                df = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - surfaced in UI
            errors.append(f"{src}: {exc}")
            df = pd.DataFrame()
        if not df.empty:
            df["__source_file__"] = os.path.basename(src)
            df["__source_path__"] = str(src)
            frames.append(df)
        if progress_cb:
            progress_cb(int(idx / total * 100))

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    artifacts = _artifacts_dir()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = artifacts / f"pcap_preview_{timestamp}.csv"
    try:
        result.to_csv(out_csv, index=False)
        out_csv_path: Optional[str] = str(out_csv)
    except Exception:
        out_csv_path = None

    result.attrs["out_csv"] = out_csv_path
    result.attrs["files_total"] = len(sources)
    result.attrs["errors"] = "\n".join(errors)
    return result
