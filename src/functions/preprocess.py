"""Vectorisation helpers used by the legacy GUI."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import pandas as pd

from PCAP import extract_sources_to_jsonl, vectorize_jsonl_files

ProgressCallback = Optional[Callable[[int], None]]


def _to_path_list(feature_source: Union[str, Sequence[str]]) -> Sequence[str]:
    if isinstance(feature_source, (list, tuple, set)):
        return [str(Path(p)) for p in feature_source]
    return [str(feature_source)]


def preprocess_feature_dir(
    feature_source: Union[str, Sequence[str]],
    out_dir: Union[str, Path],
    *,
    progress_cb: ProgressCallback = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "dataset.csv"

    sources = _to_path_list(feature_source)
    first = Path(sources[0]) if sources else None

    if not sources:
        raise FileNotFoundError("No feature sources supplied")

    if len(sources) == 1 and first and first.suffix.lower() == ".csv" and first.exists():
        shutil.copy2(first, dataset_path)
        df = pd.read_csv(dataset_path)
    elif len(sources) == 1 and first and first.suffix.lower() == ".jsonl":
        vectorize_jsonl_files([first], dataset_path, show_progress=False)
        df = pd.read_csv(dataset_path)
    elif len(sources) == 1 and first and first.exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "flows.jsonl"
            extract_sources_to_jsonl(
                first,
                jsonl_path,
                progress_callback=(lambda value: progress_cb(int(value * 0.6)) if progress_cb else None),
                show_progress=False,
            )
            vectorize_jsonl_files([jsonl_path], dataset_path, show_progress=False)
        df = pd.read_csv(dataset_path)
    else:
        frames = []
        total = len(sources)
        for idx, src in enumerate(sources, 1):
            path = Path(src)
            if path.suffix.lower() == ".csv" and path.exists():
                frames.append(pd.read_csv(path))
            elif path.suffix.lower() == ".jsonl" and path.exists():
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_csv = Path(tmpdir) / "part.csv"
                    vectorize_jsonl_files([path], tmp_csv, show_progress=False)
                    frames.append(pd.read_csv(tmp_csv))
            elif path.exists():
                with tempfile.TemporaryDirectory() as tmpdir:
                    jsonl_path = Path(tmpdir) / "flows.jsonl"
                    extract_sources_to_jsonl(path, jsonl_path, show_progress=False)
                    tmp_csv = Path(tmpdir) / "part.csv"
                    vectorize_jsonl_files([jsonl_path], tmp_csv, show_progress=False)
                    frames.append(pd.read_csv(tmp_csv))
            if progress_cb:
                progress_cb(int(idx / max(1, total) * 100))
        if not frames:
            raise FileNotFoundError("No usable feature sources")
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(dataset_path, index=False)

    if progress_cb:
        progress_cb(100)

    summary = {
        "dataset_path": str(dataset_path),
        "manifest_path": str(dataset_path),
        "meta_path": None,
        "total_rows": int(len(df)),
        "total_cols": int(len(df.columns)),
    }
    return summary
