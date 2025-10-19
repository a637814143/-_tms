#!/usr/bin/env python3
"""命令行预测脚本：对新的 PCAP 文件执行特征提取与模型推理。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from sklearn.pipeline import Pipeline

from src.configuration import get_path
from src.functions.feature_extractor import extract_features
from src.functions.unsupervised_train import META_COLUMNS, MODEL_SCHEMA_VERSION

logger = logging.getLogger("predict")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"文件 {path} 不是 JSON 对象。")
    return payload


def _resolve_feature_schema(
    metadata: Dict[str, object], feature_list_path: Optional[Path]
) -> Tuple[List[str], float, Dict[str, float]]:
    feature_order: Optional[List[str]] = None
    if feature_list_path and feature_list_path.exists():
        feature_list = _load_json(feature_list_path)
        feature_order = feature_list.get("feature_columns")
        if feature_order is None:
            raise ValueError(f"特征列表文件 {feature_list_path} 缺少 feature_columns 字段。")
        feature_hash = feature_list.get("feature_hash")
        meta_hash = metadata.get("feature_hash")
        if meta_hash and feature_hash and str(meta_hash) != str(feature_hash):
            raise ValueError(
                "特征列表与模型元数据的哈希不一致，请确认使用同一版本的数据。"
            )
    if not feature_order:
        for key in ("feature_order", "feature_columns"):
            raw = metadata.get(key)
            if raw:
                feature_order = list(raw)
                break
    if not feature_order:
        raise ValueError("模型元数据缺少特征列描述，无法进行推理。")

    default_fill = 0.0
    fill_values: Dict[str, float] = {}
    meta_fill = metadata.get("fill_values")
    if isinstance(meta_fill, dict):
        fill_values.update({str(k): float(v) for k, v in meta_fill.items()})
    preprocessor_meta = metadata.get("preprocessor")
    if isinstance(preprocessor_meta, dict):
        default_fill = float(preprocessor_meta.get("fill_value", default_fill))
        inner_fill = preprocessor_meta.get("fill_values")
        if isinstance(inner_fill, dict):
            fill_values.update({str(k): float(v) for k, v in inner_fill.items()})
    if "fill_value" in metadata:
        try:
            default_fill = float(metadata["fill_value"])
        except (TypeError, ValueError):
            pass
    return list(feature_order), float(default_fill), fill_values


def _align_features(
    df: pd.DataFrame,
    expected: List[str],
    *,
    default_fill: float,
    fill_values: Dict[str, float],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    working = df.copy()
    missing: List[str] = []
    for column in expected:
        if column in working.columns:
            continue
        fill_value = fill_values.get(column, default_fill)
        working[column] = fill_value
        missing.append(column)
    extra = [col for col in working.columns if col not in expected]
    aligned = working.loc[:, expected]
    return aligned, missing, extra


def _load_pipeline(model_path: Path):
    try:
        return joblib_load(model_path)
    except Exception as exc:
        raise RuntimeError(f"无法加载模型 {model_path}: {exc}") from exc


def _load_metadata(metadata_path: Path) -> Dict[str, object]:
    metadata = _load_json(metadata_path)
    schema_version = metadata.get("schema_version")
    if schema_version and schema_version != MODEL_SCHEMA_VERSION:
        logger.warning(
            "模型 schema_version=%s 与当前期望 %s 不一致，请确认兼容性。",
            schema_version,
            MODEL_SCHEMA_VERSION,
        )
    return metadata


def _score_predictions(
    pipeline: Pipeline,
    features: pd.DataFrame,
    metadata: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    named_steps = pipeline.named_steps
    detector = named_steps.get("detector")
    if detector is None:
        raise RuntimeError("加载的管线缺少 detector 步骤。")

    preds = pipeline.predict(features)
    is_malicious = (preds == -1).astype(int)

    pre_detector = Pipeline(pipeline.steps[:-1])
    transformed = pre_detector.transform(features)
    if transformed.ndim == 1:
        transformed = transformed.reshape(-1, 1)
    scores = detector.score_samples(transformed)
    scores = np.asarray(scores, dtype=float)

    if detector.last_vote_ratio_ is not None:
        vote_ratio = np.asarray(detector.last_vote_ratio_, dtype=float)
    elif detector.fit_votes_:
        vote_ratio = np.vstack(
            [np.where(v == -1, 1.0, 0.0) for v in detector.fit_votes_.values()]
        ).mean(axis=0)
    else:
        vote_ratio = np.ones_like(scores, dtype=float)

    score_std = float(metadata.get("score_std") or np.std(scores) or 1.0)
    threshold = float(
        detector.threshold_
        if detector.threshold_ is not None
        else metadata.get("threshold", np.quantile(scores, detector.contamination))
    )
    vote_threshold = float(
        detector.vote_threshold_
        if detector.vote_threshold_ is not None
        else metadata.get("vote_threshold", np.mean(vote_ratio))
    )

    score_component = 1.0 / (1.0 + np.exp((scores - threshold) / (score_std + 1e-6)))
    vote_component = np.clip(
        (vote_ratio - vote_threshold) / max(1e-6, (1.0 - vote_threshold)),
        0.0,
        1.0,
    )
    risk_score = np.clip(0.6 * score_component + 0.4 * vote_component, 0.0, 1.0)

    fusion_alpha = float(metadata.get("fusion_alpha", 0.5))
    if metadata.get("fusion_enabled", True):
        supervised_scores = getattr(detector, "last_supervised_scores_", None)
        calibrated_scores = getattr(detector, "last_calibrated_scores_", None)
        for extra in (supervised_scores, calibrated_scores):
            if extra is None:
                continue
            extra_arr = np.asarray(extra, dtype=float)
            if extra_arr.shape != risk_score.shape:
                continue
            risk_score = np.clip(
                fusion_alpha * risk_score + (1.0 - fusion_alpha) * np.clip(extra_arr, 0.0, 1.0),
                0.0,
                1.0,
            )
            break

    return preds, scores, vote_ratio, is_malicious, risk_score


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="对新的 PCAP 文件执行异常检测。")
    parser.add_argument("pcap", help="需要检测的 PCAP/PCAPNG 文件路径")
    parser.add_argument("--model", help="训练得到的最新管线模型路径")
    parser.add_argument("--metadata", help="模型对应的元数据 JSON")
    parser.add_argument("--feature-list", help="特征列表 JSON，用于校验列顺序")
    parser.add_argument("--output", help="输出 CSV 路径，默认写入配置的 results_prediction 目录")
    parser.add_argument("--fast", action="store_true", help="启用快速采样模式提取特征")
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="保留中间特征 CSV 文件",
    )
    args = parser.parse_args(argv)

    pcap_path = Path(args.pcap).expanduser()
    if not pcap_path.exists():
        logger.error("未找到 PCAP 文件: %s", pcap_path)
        return 1

    models_dir = get_path("models_dir")
    model_path = Path(args.model).expanduser() if args.model else models_dir / "latest_iforest_pipeline.joblib"
    if not model_path.exists():
        logger.error("模型文件不存在: %s", model_path)
        return 1

    pipeline = _load_pipeline(model_path)

    metadata_path = (
        Path(args.metadata).expanduser()
        if args.metadata
        else models_dir / "latest_iforest_metadata.json"
    )
    if not metadata_path.exists():
        logger.error("未找到模型元数据: %s", metadata_path)
        return 1
    metadata = _load_metadata(metadata_path)

    feature_list_path: Optional[Path] = None
    if args.feature_list:
        feature_list_path = Path(args.feature_list).expanduser()
    else:
        for key in ("feature_list_latest", "feature_list_path"):
            value = metadata.get(key)
            if value:
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = model_path.parent / candidate
                feature_list_path = candidate
                break
        if feature_list_path is None:
            candidate = models_dir / "latest_feature_list.json"
            if candidate.exists():
                feature_list_path = candidate

    feature_order, default_fill, fill_values = _resolve_feature_schema(metadata, feature_list_path)

    results_dir = get_path("results_prediction_dir")
    if args.output:
        output_path = Path(args.output).expanduser()
        if output_path.is_dir() or args.output.endswith("/"):
            output_dir = output_path
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"prediction_{pcap_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = output_path
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        output_file = (
            results_dir
            / f"prediction_{pcap_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        feature_csv = tmp_dir_path / f"{pcap_path.stem}_features.csv"
        logger.info("提取特征: %s -> %s", pcap_path, feature_csv)
        extract_features(str(pcap_path), str(feature_csv), fast=args.fast)

        feature_df = pd.read_csv(feature_csv, encoding="utf-8")
        if feature_df.empty:
            logger.error("特征文件为空，无法预测。")
            return 1

        meta_cols = [col for col in feature_df.columns if col in META_COLUMNS]
        feature_values = feature_df.drop(columns=meta_cols, errors="ignore")
        aligned_features, missing_cols, extra_cols = _align_features(
            feature_values,
            feature_order,
            default_fill=default_fill,
            fill_values=fill_values,
        )

        raw_preds, scores, vote_ratio, is_malicious, risk_score = _score_predictions(
            pipeline,
            aligned_features,
            metadata,
        )

        out_df = feature_df.copy()
        out_df["prediction"] = raw_preds
        out_df["is_malicious"] = is_malicious
        out_df["anomaly_score"] = scores
        out_df["vote_ratio"] = vote_ratio
        out_df["risk_score"] = risk_score
        out_df["anomaly_confidence"] = risk_score

        output_file.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_file, index=False, encoding="utf-8")

        malicious = int(is_malicious.sum())
        total = int(len(out_df))
        logger.info(
            "预测完成：结果=%s 异常=%d/%d (%.2f%%) 输出=%s",
            "异常" if malicious else "正常",
            malicious,
            total,
            (malicious / total * 100.0) if total else 0.0,
            output_file,
        )
        if missing_cols:
            logger.warning("输入缺少列，已按默认值补齐: %s", ", ".join(missing_cols))
        if extra_cols:
            logger.info("输入包含未使用列，已忽略: %s", ", ".join(extra_cols))

        if args.keep_intermediate:
            keep_path = output_file.parent / f"{output_file.stem}_features.csv"
            feature_csv.replace(keep_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
