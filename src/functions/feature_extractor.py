"""PCAP feature extraction orchestrator mirroring the PE static pipeline."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

try:  # Optional dependency – fall back to ``csv`` writer if unavailable.
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional at runtime
    pd = None  # type: ignore

from .annotations import configured_plugin_dirs
from .static_features import (
    ExtractionSummary as _StaticExtractionSummary,
    ThreadSafeFileWriter,
    ThreadSafeProgressTracker,
    extract_pcap_features,
    extract_pcap_features_batch,
    extract_pcap_features_to_file,
    extract_sources_to_jsonl,
    list_pcap_sources,
)

__all__ = [
    "ThreadSafeProgressTracker",
    "ThreadSafeFileWriter",
    "extract_pcap_features",
    "extract_pcap_features_batch",
    "extract_pcap_features_to_file",
    "extract_sources_to_jsonl",
    "list_pcap_sources",
    "extract_features",
    "extract_features_dir",
    "get_loaded_plugin_info",
    "ExtractionSummary",
]

_PCAP_SUFFIXES: Tuple[str, ...] = (".pcap", ".pcapng")
_CSV_ENCODING = "utf-8"
_META_ALIASES: Tuple[str, ...] = ()


ExtractionSummary = _StaticExtractionSummary


@lru_cache(maxsize=1)
def _ordered_flow_columns() -> Tuple[str, ...]:
    """Return the canonical CSV column order when available."""

    try:
        from .vectorizer import CSV_COLUMNS  # circular import safe at runtime
    except Exception:  # pragma: no cover - vectorizer may be unavailable
        return ()
    return tuple(CSV_COLUMNS)


def _augment_flow_record(record: Dict[str, object], _source: Path) -> Dict[str, object]:
    """Ensure required canonical columns are present for downstream users."""

    payload = dict(record)
    payload.setdefault("Flow ID", payload.get("Flow ID", ""))
    payload.setdefault("Source IP", payload.get("Source IP", ""))
    payload.setdefault("Destination IP", payload.get("Destination IP", ""))
    payload.setdefault("Source Port", payload.get("Source Port", 0))
    payload.setdefault("Destination Port", payload.get("Destination Port", 0))
    payload.setdefault("Protocol", payload.get("Protocol", 0))
    payload.setdefault("Label", payload.get("Label", 0))

    return payload


def _column_order(existing: Iterable[str]) -> List[str]:
    """Compute a stable column order honouring the canonical header."""

    desired = list(_ordered_flow_columns())
    seen: set[str] = set()
    order: List[str] = []

    for column in desired:
        if column in existing and column not in seen:
            order.append(column)
            seen.add(column)

    for alias in _META_ALIASES:
        if alias in existing and alias not in seen:
            order.append(alias)
            seen.add(alias)

    for column in existing:
        if column not in seen:
            order.append(column)
            seen.add(column)

    return order


def _write_flow_csv(output_path: Path, rows: List[Dict[str, object]]) -> None:
    """Persist flow records to disk using pandas when available."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        columns = list(_ordered_flow_columns())
        for alias in _META_ALIASES:
            if alias not in columns:
                columns.append(alias)
        if not columns and rows:
            columns = list(rows[0].keys())
        with output_path.open("w", encoding=_CSV_ENCODING, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
        return

    if pd is not None:
        frame = pd.DataFrame(rows)
        order = _column_order(frame.columns)
        frame.to_csv(output_path, index=False, columns=order, encoding=_CSV_ENCODING)
        return

    order = _column_order(rows[0].keys())
    with output_path.open("w", encoding=_CSV_ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=order)
        writer.writeheader()
        writer.writerows(rows)


def extract_features(
    pcap_path: str,
    output_csv: str,
    packet_index: Optional[int] = None,
    progress_cb=None,
) -> str:
    """Extract flow features and export them to a CSV file.

    The ``packet_index`` argument is accepted for backwards compatibility but
    currently ignored – the modern extractor always exports all flows for the
    given PCAP file.
    """

    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"pcap 不存在: {pcap_path}")

    result = extract_pcap_features(path)
    if not result.get("success", False):
        raise RuntimeError(result.get("error", "特征提取失败"))

    rows = [
        _augment_flow_record(record, path)
        for record in result.get("flows", [])
    ]

    _write_flow_csv(Path(output_csv), rows)

    if progress_cb:
        progress_cb(100)

    return str(Path(output_csv).resolve())


def extract_features_dir(
    split_dir: str,
    out_dir: str,
    workers: int = 4,
    progress_cb=None,
) -> List[str]:
    """Batch extract features for every PCAP/PCAPNG inside ``split_dir``."""

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
        if progress_cb:
            pct = int(completed / total * 100) if total else 100
            progress_cb(min(100, max(0, pct)))

    def _process(path: Path) -> str:
        target = output_dir / f"{path.stem}_features.csv"
        return extract_features(str(path), str(target), progress_cb=None)

    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process, path): path for path in inputs}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(f"特征提取失败: {futures[future]} ({exc})") from exc
                results.append(result)
                completed += 1
                _notify()
    else:
        for path in inputs:
            result = _process(path)
            results.append(result)
            completed += 1
            _notify()

    results.sort()
    return results


def get_loaded_plugin_info() -> List[Dict[str, object]]:
    """Return a lightweight summary of available feature plugins."""

    info: List[Dict[str, object]] = []

    for directory in configured_plugin_dirs():
        if not directory.exists():
            continue
        for path in sorted(directory.glob("**/*.py")):
            if path.name.startswith("_") or path.name == "__init__.py":
                continue
            module_name = ".".join(path.relative_to(directory).with_suffix("").parts)
            info.append({
                "module": module_name,
                "extractors": [],
            })

    if not info:
        info.append(
            {
                "module": "builtin.feature_extractor",
                "extractors": [
                    "extract_features",
                    "extract_features_dir",
                    "extract_pcap_features",
                ],
            }
        )

    return info