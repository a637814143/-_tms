"""Command line utilities for PCAP feature extraction and ML workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .modeling import detect_pcap_with_model, train_hist_gradient_boosting
from .static_features import extract_sources_to_jsonl
from .vectorizer import vectorize_jsonl_files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract", help="Extract PCAP/PCAPNG features into JSONL format"
    )
    extract_parser.add_argument("source", type=Path, help="PCAP file or directory to process")
    extract_parser.add_argument("output", type=Path, help="Destination JSONL file")
    extract_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional worker count for parallel extraction",
    )

    vec_parser = subparsers.add_parser("vectorize", help="Vectorize JSONL feature files")
    vec_parser.add_argument("output", type=Path, help="Destination CSV file")
    vec_parser.add_argument("jsonl", nargs="+", type=Path, help="JSONL feature files")
    vec_parser.add_argument(
        "--label",
        type=int,
        default=None,
        help="Optional label applied to all provided flows",
    )

    train_parser = subparsers.add_parser("train", help="Train a tree-based classifier")
    train_parser.add_argument("dataset", type=Path, help="Vectorized dataset (.csv)")
    train_parser.add_argument("model", type=Path, help="Output model path (model.joblib)")
    train_parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override the default number of boosting iterations",
    )

    detect_parser = subparsers.add_parser("detect", help="Run inference on a PCAP file")
    detect_parser.add_argument("model", type=Path, help="Trained model path")
    detect_parser.add_argument("pcap", type=Path, help="PCAP file to analyse")

    return parser


def _handle_extract(source: Path, output: Path, workers: int | None) -> None:
    summary = extract_sources_to_jsonl(
        source,
        output,
        max_workers=workers,
        show_progress=True,
    )
    print(
        "Extracted features for {count} files (success: {success}) into {path}".format(
            count=summary.record_count,
            success=summary.success_count,
            path=summary.path,
        )
    )
    if summary.success_count < summary.record_count:
        failures = summary.record_count - summary.success_count
        print(f"Warning: {failures} PCAP files failed during extraction.")


def _handle_vectorize(output: Path, jsonl_paths: Sequence[Path], label: int | None) -> None:
    summary = vectorize_jsonl_files(
        jsonl_paths,
        output,
        label_override=label,
        show_progress=True,
    )
    print(f"Saved dataset to {summary.path} with {summary.flow_count} flows")
    label_note = "with labels" if summary.has_labels else "without labels"
    print(f"CSV includes {summary.column_count} columns {label_note}.")


def _handle_train(dataset: Path, model: Path, iterations: int | None) -> None:
    kwargs = {}
    if iterations is not None:
        kwargs["max_iter"] = iterations
    summary = train_hist_gradient_boosting(dataset, model, **kwargs)
    print(
        "Model trained on {flows} flows across {features} features. Classes: {classes}. Saved to {path}".format(
            flows=summary.flow_count,
            features=len(summary.feature_names),
            classes=summary.classes,
            path=summary.model_path,
        )
    )
    if summary.label_mapping:
        print(f"Label mapping: {summary.label_mapping}")
    if summary.dropped_flows:
        print(f"Dropped {summary.dropped_flows} unlabeled rows prior to training.")


def _handle_detect(model: Path, pcap: Path) -> None:
    result = detect_pcap_with_model(model, pcap)
    if not result.success:
        raise SystemExit(f"Detection failed: {result.error}")

    if result.fusion_flags is not None:
        suspicious = sum(1 for flag in result.fusion_flags if bool(flag))
    else:
        suspicious = sum(1 for value in result.predictions if value != 0)

    print(
        "Analysed {flows} flows. Suspicious flows: {suspicious}.".format(
            flows=result.flow_count,
            suspicious=suspicious,
        )
    )
    if result.scores:
        top_score = max(result.scores)
        print(f"Highest malicious score: {top_score:.3f}")
    if result.prediction_labels:
        preview = result.prediction_labels[:5]
        print(f"Predicted labels (first 5): {preview}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "extract":
        _handle_extract(args.source, args.output, args.workers)
    elif args.command == "vectorize":
        _handle_vectorize(args.output, args.jsonl, args.label)
    elif args.command == "train":
        _handle_train(args.dataset, args.model, args.iterations)
    elif args.command == "detect":
        _handle_detect(args.model, args.pcap)
    else:  # pragma: no cover - defensive
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
