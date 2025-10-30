"""CLI compatibility layer exposing the new PCAP workflow to the legacy GUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import load

from PCAP import load_vectorized_dataset


def _predict_from_features(pipeline_path: Path, feature_csv: Path, output_path: Optional[Path]) -> Path:
    artifact = load(pipeline_path)
    model = artifact.get("model")
    label_mapping = artifact.get("label_mapping")
    if model is None:
        raise RuntimeError("模型文件缺少必要信息")

    X, _, _, _, _ = load_vectorized_dataset(feature_csv, show_progress=False, return_stats=True)
    if X.size == 0:
        raise RuntimeError("特征 CSV 为空，无法预测")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        scores = proba.max(axis=1)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if decision.ndim == 1:
            scores = 1.0 / (1.0 + np.exp(-decision))
        else:
            scores = decision.max(axis=1)
    else:
        scores = model.predict(X)

    preds = model.predict(X)
    if label_mapping:
        labels = [label_mapping.get(int(val), str(val)) for val in preds]
    else:
        labels = [str(val) for val in preds]

    df = pd.read_csv(feature_csv)
    df["prediction"] = preds
    df["prediction_label"] = labels
    df["malicious_score"] = scores

    if output_path is None:
        output_path = feature_csv.with_name(f"{feature_csv.stem}_prediction.csv")
    df.to_csv(output_path, index=False)

    if label_mapping and 0 in label_mapping:
        benign_label = str(label_mapping[0])
        malicious_count = int((df["prediction_label"].astype(str) != benign_label).sum())
    else:
        benign_label = labels[0] if labels else None
        malicious_count = int((df["prediction_label"].astype(str) != str(benign_label)).sum()) if benign_label else 0

    summary = {
        "total": int(len(df)),
        "malicious": malicious_count,
    }
    output_json = output_path.with_suffix(".json")
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_path


def cmd_predict(args: argparse.Namespace) -> int:
    pipeline = Path(args.pipeline)
    features = Path(args.features)
    output = Path(args.output) if args.output else None
    result = _predict_from_features(pipeline, features, output)
    print(str(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCAP workflow CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="使用训练好的模型对特征 CSV 进行预测")
    predict.add_argument("pipeline", help="模型 joblib 路径")
    predict.add_argument("features", help="特征 CSV 路径")
    predict.add_argument("--metadata", help="兼容参数，占位不使用")
    predict.add_argument("--output", help="预测结果 CSV")
    predict.set_defaults(func=cmd_predict)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
