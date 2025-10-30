"""PCAP feature extraction orchestrator mirroring the PE static pipeline."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .feature_utils import extract_flow_features

try:  # Optional dependency used for user-facing progress bars.
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None  # type: ignore

__all__ = [
    "ThreadSafeProgressTracker",
    "ThreadSafeFileWriter",
    "extract_pcap_features",
    "extract_pcap_features_batch",
    "extract_pcap_features_to_file",
    "extract_sources_to_jsonl",
    "list_pcap_sources",
    "ExtractionSummary",
]

_PCAP_SUFFIXES: Tuple[str, ...] = (".pcap", ".pcapng")


@dataclass
class ExtractionSummary:
    """Summary describing a JSONL export produced from PCAP files."""

    path: Path
    source_count: int
    record_count: int
    success_count: int


class ThreadSafeProgressTracker:
    """Thread-safe helper used to surface extraction progress."""

    def __init__(self, total_items: int, progress_callback=None, text_callback=None):
        self.total_items = max(total_items, 1)
        self.progress_callback = progress_callback or (lambda *_: None)
        self.text_callback = text_callback or (lambda *_: None)
        self.completed_items = 0
        self.lock = threading.Lock()

    def update_progress(self, file_progress: float, file_name: str = "") -> None:
        with self.lock:
            overall_progress = (self.completed_items + file_progress / 100.0) / self.total_items * 100.0
            self.progress_callback(int(overall_progress))
            if file_name:
                self.text_callback(f"Processing {file_name} ({file_progress:.1f}%)")

    def complete_item(self, file_name: str = "") -> None:
        with self.lock:
            self.completed_items += 1
            overall_progress = self.completed_items / self.total_items * 100.0
            self.progress_callback(int(overall_progress))
            if file_name:
                self.text_callback(f"Completed {file_name} ({self.completed_items}/{self.total_items})")


class ThreadSafeFileWriter:
    """Streaming writer for JSON lines output."""

    def __init__(self, file_path: Path, text_callback=None):
        self.file_path = file_path
        self.text_callback = text_callback or (lambda *_: None)
        self.lock = threading.Lock()
        self.file_handle = open(file_path, "w", encoding="utf-8", buffering=1)
        self.written = 0

    def write_result(self, result: Dict[str, object]) -> None:
        with self.lock:
            self.file_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            self.file_handle.flush()
            self.written += 1

    def close(self) -> None:
        with self.lock:
            if getattr(self, "file_handle", None):
                self.file_handle.close()
                self.file_handle = None
                self.text_callback(f"Finished writing {self.written} records to {self.file_path}")

    def __del__(self):
        self.close()


def extract_pcap_features(pcap_path: Union[str, Path], progress_callback=None) -> Dict[str, object]:
    """Extract CIC-style flow statistics for a single PCAP file."""

    pcap_path = Path(pcap_path)
    progress_callback = progress_callback or (lambda *_: None)

    try:
        flows = extract_flow_features(pcap_path)
        progress_callback(100)
        return {
            "success": True,
            "path": str(pcap_path),
            "flows": flows,
        }
    except Exception as exc:  # pragma: no cover - defensive by design
        progress_callback(100)
        return {
            "success": False,
            "path": str(pcap_path),
            "error": str(exc),
            "flows": [],
        }


def extract_pcap_features_batch(
    inputs: Iterable[Union[str, Path]],
    *,
    max_workers: Optional[int] = None,
    progress_callback=None,
    text_callback=None,
) -> List[Dict[str, object]]:
    """Parallel feature extraction over multiple PCAP files."""

    paths = [Path(item) for item in inputs]
    tracker = ThreadSafeProgressTracker(len(paths), progress_callback, text_callback)
    results: List[Dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for path in paths:
            callback = lambda progress, name=path.name: tracker.update_progress(progress, name)
            future = executor.submit(extract_pcap_features, path, callback)
            future_map[future] = path
        for future in as_completed(future_map):
            path = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                result = {"success": False, "path": str(path), "error": str(exc), "flows": []}
            results.append(result)
            tracker.complete_item(path.name)

    return results


def list_pcap_sources(source: Union[str, Path]) -> List[Path]:
    """Resolve a PCAP/PCAPNG path (file or directory) into concrete files."""

    path = Path(source)

    if path.is_dir():
        files = [
            item
            for item in sorted(path.rglob("*"))
            if item.is_file() and item.suffix.lower() in _PCAP_SUFFIXES
        ]
        if not files:
            raise FileNotFoundError(f"No PCAP/PCAPNG files found in {path}")
        return files

    if path.is_file() and path.suffix.lower() in _PCAP_SUFFIXES:
        return [path]

    raise FileNotFoundError(f"Unsupported source for PCAP extraction: {path}")


def _write_results_to_jsonl(
    inputs: Sequence[Path],
    output_file: Union[str, Path],
    *,
    max_workers: Optional[int] = None,
    progress_callback=None,
    text_callback=None,
    show_progress: bool = False,
) -> ExtractionSummary:
    """Helper that streams extraction results to a JSONL file."""

    output_path = Path(output_file)
    writer = ThreadSafeFileWriter(output_path, text_callback)
    total_records = 0
    success_count = 0

    progress = (
        tqdm(total=len(inputs), desc="Extracting PCAPs", unit="file", leave=False)
        if show_progress and tqdm is not None
        else None
    )

    try:
        for result in extract_pcap_features_batch(
            inputs,
            max_workers=max_workers,
            progress_callback=progress_callback,
            text_callback=text_callback,
        ):
            writer.write_result(result)
            total_records += 1
            if result.get("success", False):
                success_count += 1
            if progress is not None:
                progress.update(1)
    finally:
        writer.close()
        if progress is not None:
            progress.close()

    return ExtractionSummary(
        path=output_path,
        source_count=len(inputs),
        record_count=total_records,
        success_count=success_count,
    )


def extract_pcap_features_to_file(
    inputs: Iterable[Union[str, Path]],
    output_file: Union[str, Path],
    *,
    max_workers: Optional[int] = None,
    progress_callback=None,
    text_callback=None,
    show_progress: bool = False,
) -> Path:
    """Extract features for multiple PCAP files and write them as JSON lines."""

    paths = [Path(item) for item in inputs]
    summary = _write_results_to_jsonl(
        paths,
        output_file,
        max_workers=max_workers,
        progress_callback=progress_callback,
        text_callback=text_callback,
        show_progress=show_progress,
    )
    return summary.path


def extract_sources_to_jsonl(
    source: Union[str, Path],
    output_file: Union[str, Path],
    *,
    max_workers: Optional[int] = None,
    progress_callback=None,
    text_callback=None,
    show_progress: bool = False,
) -> ExtractionSummary:
    """Extract features from a path or PCAP file directly into JSONL output."""

    inputs = list_pcap_sources(source)
    return _write_results_to_jsonl(
        inputs,
        output_file,
        max_workers=max_workers,
        progress_callback=progress_callback,
        text_callback=text_callback,
        show_progress=show_progress,
    )
