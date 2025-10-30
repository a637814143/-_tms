"""Feature extraction helpers that proxy to the PCAP workflow."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from PCAP import extract_sources_to_jsonl, vectorize_jsonl_files

ProgressCallback = Optional[Callable[[int], None]]


def _iter_pcap_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    patterns = ("*.pcap", "*.pcapng")
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(root.rglob(pattern))
    return sorted({p.resolve() for p in matches if p.is_file()})


def extract_features(
    pcap_path: str,
    csv_path: str,
    *,
    label: Optional[int] = None,
    workers: Optional[int] = None,
    progress_cb: ProgressCallback = None,
    **_: object,
) -> str:
    """Extract a single PCAP file into a feature CSV."""

    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    source = Path(pcap_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "flows.jsonl"
        extract_sources_to_jsonl(
            source,
            jsonl_path,
            max_workers=workers,
            progress_callback=(lambda value: progress_cb(int(value * 0.6)) if progress_cb else None),
            show_progress=False,
        )
        vectorize_jsonl_files(
            [jsonl_path],
            csv_file,
            label_override=label,
            show_progress=False,
        )

    if progress_cb:
        progress_cb(100)
    return str(csv_file)


def extract_features_dir(
    split_dir: str,
    out_dir: str,
    *,
    workers: Optional[int] = None,
    progress_cb: ProgressCallback = None,
    **_: object,
) -> List[str]:
    """Extract every PCAP under ``split_dir`` into individual CSV files."""

    root = Path(split_dir)
    dest_dir = Path(out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_pcap_files(root)
    if not files:
        raise FileNotFoundError(f"No PCAP/PCAPNG files found under {split_dir}")

    outputs: List[str] = []
    total = len(files)
    for idx, file_path in enumerate(files, 1):
        csv_name = file_path.with_suffix(".csv").name
        csv_file = dest_dir / csv_name
        extract_features(
            str(file_path),
            str(csv_file),
            workers=workers,
            progress_cb=None,
        )
        outputs.append(str(csv_file))
        if progress_cb:
            progress_cb(int(idx / total * 100))

    if progress_cb:
        progress_cb(100)
    return outputs


def get_loaded_plugin_info() -> List[dict]:
    """Return metadata describing the active PCAP extraction pipeline."""

    return [
        {
            "module": "PCAP.static_features",
            "extractors": [
                "PCAP flow feature extractor (JSONL)",
            ],
        },
        {
            "module": "PCAP.vectorizer",
            "extractors": [
                "CSV vectorizer compatible with EMBER-style models",
            ],
        },
    ]
