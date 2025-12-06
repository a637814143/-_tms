"""PCAP feature extraction orchestrator mirroring the reference pipeline."""

from __future__ import annotations

import csv
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from .feature_utils import extract_flow_features

try:  # Optional dependency used for user-facing progress bars.
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None  # type: ignore

__all__ = [
    "ThreadSafeProgressTracker",
    "ThreadSafeFileWriter",
    "ExtractionSummary",
    "extract_pcap_features",
    "extract_pcap_features_batch",
    "extract_pcap_features_to_file",
    "extract_sources_to_jsonl",
    "list_pcap_sources",
    "extract_features",
    "extract_features_dir",
    "get_loaded_plugin_info",
]

_PCAP_SUFFIXES: Tuple[str, ...] = (".pcap", ".pcapng")
_LABEL_COLUMN = "Label"
_DUPLICATE_COLUMN_ALIASES: Dict[Tuple[str, int], str] = {("Fwd Header Length", 1): "Fwd Header Length.1"}


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

    def __del__(self):  # pragma: no cover - defensive close
        self.close()


def extract_pcap_features(
    pcap_path: Union[str, Path],
    progress_callback=None,
    *,
    fast: Optional[bool] = None,
) -> Dict[str, object]:
    """Extract CIC-style flow statistics for a single PCAP file."""

    # ``fast`` is accepted for compatibility with previous releases but ignored.
    _ = fast

    pcap_path = Path(pcap_path)
    progress_callback = progress_callback or (lambda *_: None)

    try:
        packet_stats: Dict[str, int] = {}
        flows = extract_flow_features(pcap_path, stats=packet_stats)
        warnings: List[str] = []

        if packet_stats.get("parsed_packets", 0) == 0:
            warnings.append("未能从 PCAP 中解析出任何数据包，文件可能为空或格式不兼容。")
        elif packet_stats.get("dropped_packets", 0):
            warnings.append(
                f"有 {packet_stats['dropped_packets']} 个数据包在解析时被跳过，预测可靠性可能受影响。"
            )

        progress_callback(100)
        return {
            "success": True,
            "path": str(pcap_path),
            "flows": flows,
            "packet_stats": packet_stats,
            "warnings": warnings,
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
    fast: Optional[bool] = None,
) -> List[Dict[str, object]]:
    """Parallel feature extraction over multiple PCAP files."""

    # ``fast`` flag kept for backwards compatibility, but the implementation always returns full features.
    _ = fast

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
    fast: Optional[bool] = None,
) -> Path:
    """Extract features for multiple PCAP files and write them as JSON lines."""

    _ = fast

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
    fast: Optional[bool] = None,
) -> ExtractionSummary:
    """Extract features from a path or PCAP file directly into JSONL output."""

    _ = fast

    inputs = list_pcap_sources(source)
    return _write_results_to_jsonl(
        inputs,
        output_file,
        max_workers=max_workers,
        progress_callback=progress_callback,
        text_callback=text_callback,
        show_progress=show_progress,
    )


def _resolve_actual_key(column: str, occurrence: int) -> str:
    return _DUPLICATE_COLUMN_ALIASES.get((column, occurrence), column)


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _flows_to_csv(flows: Iterable[Dict[str, object]], output_csv: Union[str, Path]) -> Path:
    from .vectorizer import CSV_COLUMNS

    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for flow in flows:
            counts: Dict[str, int] = defaultdict(int)
            row: List[str] = []
            for column in CSV_COLUMNS:
                occurrence = counts[column]
                counts[column] += 1
                if column == _LABEL_COLUMN:
                    value = flow.get(_LABEL_COLUMN)
                else:
                    key = _resolve_actual_key(column, occurrence)
                    value = flow.get(key)
                row.append(_format_value(value))
            writer.writerow(row)

    return path


def extract_features(
    pcap_path: str,
    output_csv: str,
    packet_index: Optional[int] = None,
    progress_cb=None,
    *,
    fast: bool = False,
    **kwargs,
) -> str:
    """Extract flow features and export them to a CSV file."""

    _ = packet_index
    _ = kwargs
    _ = fast  # preserved for compatibility; extraction always returns the full feature set.

    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"pcap 不存在: {pcap_path}")

    result = extract_pcap_features(path, progress_cb)
    if not result.get("success", False):
        raise RuntimeError(result.get("error", "Feature extraction failed"))

    output_path = _flows_to_csv(result.get("flows", []), output_csv)
    if progress_cb:
        progress_cb(100)
    return str(output_path.resolve())


def extract_features_dir(
    split_dir: str,
    out_dir: str,
    workers: int = 4,
    progress_cb=None,
    *,
    fast: bool = False,
    **kwargs,
) -> List[str]:
    """Batch extract features for every PCAP/PCAPNG inside ``split_dir``."""

    _ = kwargs
    _ = fast

    directory = Path(split_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在: {split_dir}")

    inputs = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in _PCAP_SUFFIXES
    ]
    if not inputs:
        raise RuntimeError(f"目录下无 pcap: {split_dir}")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[str] = []
    total = len(inputs)
    completed = 0

    def _notify() -> None:
        nonlocal completed
        if progress_cb:
            pct = int(completed / total * 100) if total else 100
            progress_cb(min(100, max(0, pct)))

    def _process(path: Path) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = output_dir / f"{path.stem}_features_{timestamp}.csv"
        result = extract_pcap_features(path)
        if not result.get("success", False):
            raise RuntimeError(result.get("error", "Feature extraction failed"))
        return str(_flows_to_csv(result.get("flows", []), target))

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process, path): path for path in inputs}
            for future in as_completed(futures):
                try:
                    result_path = future.result()
                except Exception as exc:
                    raise RuntimeError(f"特征提取失败: {futures[future]} ({exc})") from exc
                results.append(result_path)
                completed += 1
                _notify()
    else:
        for path in inputs:
            result_path = _process(path)
            results.append(result_path)
            completed += 1
            _notify()

    results.sort()
    return results


def get_loaded_plugin_info() -> List[Dict[str, object]]:
    """Return a lightweight summary of available feature extractors."""

    return [
        {
            "module": "builtin.feature_extractor",
            "extractors": [
                "extract_features",
                "extract_features_dir",
                "extract_pcap_features",
            ],
        }
    ]