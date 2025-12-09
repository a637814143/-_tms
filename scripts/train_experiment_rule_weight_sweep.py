"""Scan different model/rule weight pairs and summarise fusion metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.configuration import load_config
from src.functions.modeling import train_supervised_on_split
from src.functions.risk_rules import fuse_model_rule_votes


def _load_config(path: Path) -> Dict[str, object]:
    cfg = load_config(str(path))
    return cfg if isinstance(cfg, dict) else {}


def _parse_weights(raw: str) -> List[Tuple[float, float]]:
    combos: List[Tuple[float, float]] = []
    for pair in raw.split(";"):
        parts = pair.split(",")
        if len(parts) != 2:
            continue
        try:
            model_w = float(parts[0])
            rule_w = float(parts[1])
        except ValueError:
            continue
        combos.append((model_w, rule_w))
    return combos or [(0.6, 0.4)]


def _load_dataset(train_path: Path, feature_names: Iterable[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    if train_path.is_dir():
        frames = [pd.read_csv(p) for p in sorted(train_path.rglob("*.csv"))]
        if not frames:
            raise FileNotFoundError(f"目录 {train_path} 中没有找到 CSV 文件。")
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(train_path)

    feature_df = df[list(feature_names)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    label_cols = [col for col in df.columns if col.lower() == "labelbinary"]
    if not label_cols:
        raise ValueError("训练数据缺少 LabelBinary 列，无法计算指标。")
    labels = df[label_cols[0]]
    y = labels.to_numpy(dtype=np.int64)
    return feature_df, y


def _extract_model_scores(model, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        if proba.shape[1] > 1:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(features)
        return 1.0 / (1.0 + np.exp(-decision))
    preds = model.predict(features)
    return np.asarray(preds, dtype=np.float64)


def _compute_metrics(flags: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, flags)),
        "precision": float(precision_score(labels, flags, zero_division=0)),
        "recall": float(recall_score(labels, flags, zero_division=0)),
        "f1": float(f1_score(labels, flags, zero_division=0)),
        "auc": float(roc_auc_score(labels, flags)) if len(np.unique(labels)) > 1 else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="比较不同模型/规则权重组合的融合效果。")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config/default.yaml"),
        help="配置文件路径，默认使用 config/default.yaml",
    )
    parser.add_argument(
        "--train-csv",
        required=True,
        help="带标签的训练 CSV 或目录路径。",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="指标输出目录。",
    )
    parser.add_argument(
        "--weights",
        default="0.8,0.2;0.6,0.4;0.5,0.5",
        help="模型权重与规则权重组合，使用分号分隔，例如 '1.0,0.0;0.5,0.5'。",
    )
    parser.add_argument(
        "--fusion-threshold",
        type=float,
        default=0.5,
        help="融合判定阈值，默认 0.5。",
    )
    parser.add_argument(
        "--trigger-threshold",
        type=float,
        default=60.0,
        help="规则触发阈值，默认 60。",
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_csv)

    # 先训练一次基线模型，用于后续不同权重的融合实验
    results = train_supervised_on_split(
        split_dir=train_path,
        results_dir=output_dir,
        models_dir=output_dir,
        model_tag="rule_weight_sweep",
        **(config.get("training") or {}),
    )

    pipeline = Path(results["model_path"])
    payload = joblib.load(pipeline)
    model = payload["model"]
    feature_names = payload["feature_names"]

    features, labels = _load_dataset(train_path, feature_names)
    model_scores = _extract_model_scores(model, features)

    combos = _parse_weights(args.weights)
    records: List[Dict[str, object]] = []

    rule_scores = model_scores * 100.0  # 使用模型得分近似模拟规则触发程度

    for model_w, rule_w in combos:
        fused_scores, fused_flags, rules_triggered = fuse_model_rule_votes(
            model_scores,
            rule_scores,
            model_weight=model_w,
            rule_weight=rule_w,
            threshold=args.fusion_threshold,
            trigger_threshold=args.trigger_threshold,
        )
        metrics = _compute_metrics(fused_flags, labels)
        records.append(
            {
                "model_weight": model_w,
                "rule_weight": rule_w,
                **metrics,
                "rules_triggered": int(rules_triggered.sum()),
                "fused_positive": int(fused_flags.sum()),
            }
        )

    output_csv = output_dir / "rule_weight_metrics.csv"
    pd.DataFrame(records).to_csv(output_csv, index=False)

    print(f"权重扫描完成，结果写入 {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())