
"""CLI and optional REST services for malware detector workflows."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:  # Optional heavy dependencies. Provide graceful degradation when absent.
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback path
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback path
    pd = None  # type: ignore

try:
    from joblib import load as joblib_load  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback path
    joblib_load = None  # type: ignore

try:
    from src.functions.analyze_results import analyze_results
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    analyze_results = None  # type: ignore

try:
    from src.functions.feature_extractor import extract_features_dir
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    extract_features_dir = None  # type: ignore
from src.functions.logging_utils import get_logger, log_model_event
from src.functions.csv_utils import read_csv_flexible
try:
    from src.functions.modeling import ModelTrainer, summarize_prediction_labels
except ModuleNotFoundError:  # pragma: no cover - triggered in lightweight envs
    ModelTrainer = None  # type: ignore[assignment]

    def summarize_prediction_labels(
        predictions: Iterable[object],
        label_mapping: Optional[Dict[int, str]] = None,
    ) -> Tuple[List[str], int, int, Optional[str]]:
        labels: List[str] = []
        anomaly_count = 0
        for value in predictions:
            mapped = None
            if label_mapping is not None:
                try:
                    mapped = label_mapping.get(int(value))
                except (TypeError, ValueError):
                    mapped = None
            label = mapped or str(value)
            labels.append(label)
            try:
                ivalue = int(value)
            except (TypeError, ValueError):
                continue
            if ivalue in (1, -1):
                anomaly_count += 1
        normal_count = max(len(labels) - anomaly_count, 0)
        status = "异常" if anomaly_count > 0 else ("正常" if labels else None)
        return labels, anomaly_count, normal_count, status

try:
    from src.functions.risk_rules import (  # type: ignore
        fuse_model_rule_votes,
        get_fusion_settings,
        get_rule_settings,
        score_rules as apply_risk_rules,
        DEFAULT_TRIGGER_THRESHOLD,
        DEFAULT_MODEL_WEIGHT,
        DEFAULT_RULE_WEIGHT,
        DEFAULT_FUSION_THRESHOLD,
        RULE_TRIGGER_THRESHOLD,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    apply_risk_rules = None  # type: ignore
    DEFAULT_TRIGGER_THRESHOLD = 40.0  # type: ignore
    DEFAULT_MODEL_WEIGHT = 0.3  # type: ignore
    DEFAULT_RULE_WEIGHT = 0.7  # type: ignore
    DEFAULT_FUSION_THRESHOLD = 0.2  # type: ignore
    RULE_TRIGGER_THRESHOLD = 25.0  # type: ignore

    def get_rule_settings(profile: Optional[str] = None) -> Dict[str, object]:  # type: ignore
        return {
            "params": {},
            "trigger_threshold": float(DEFAULT_TRIGGER_THRESHOLD),
            "model_weight": float(DEFAULT_MODEL_WEIGHT),
            "rule_weight": float(DEFAULT_RULE_WEIGHT),
            "fusion_threshold": float(DEFAULT_FUSION_THRESHOLD),
            "profile": profile,
        }

    def get_fusion_settings(profile: Optional[str] = None) -> Dict[str, float]:  # type: ignore
        return {
            "model_weight": float(DEFAULT_MODEL_WEIGHT),
            "rule_weight": float(DEFAULT_RULE_WEIGHT),
            "fusion_threshold": float(DEFAULT_FUSION_THRESHOLD),
            "profile": profile or "baseline",
        }

    def fuse_model_rule_votes(  # type: ignore
        model_scores: Iterable[object],
        rule_scores: Optional[Iterable[object]],
        *,
        profile: Optional[str] = None,
        model_weight: Optional[float] = None,
        rule_weight: Optional[float] = None,
        threshold: Optional[float] = None,
        trigger_threshold: Optional[float] = None,
        model_confidence: Optional[Iterable[object]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model_arr = np.asarray(model_scores, dtype=np.float64).reshape(-1)
        if model_arr.size == 0:
            empty = np.zeros(0, dtype=np.float64)
            empty_bool = empty.astype(bool)
            return empty, empty_bool, empty_bool

        model_arr = np.clip(model_arr, 0.0, 1.0)

        if rule_scores is None:
            rule_arr = np.zeros_like(model_arr)
            normalized_rules = rule_arr
            active_rule_weight = float(rule_weight or 0.0)
        else:
            rule_arr = np.asarray(rule_scores, dtype=np.float64).reshape(-1)
            if rule_arr.size != model_arr.size:
                if rule_arr.size < model_arr.size:
                    padded = np.zeros_like(model_arr)
                    padded[: rule_arr.size] = rule_arr
                    rule_arr = padded
                else:
                    rule_arr = rule_arr[: model_arr.size]
            normalized_rules = np.clip(rule_arr / 100.0, 0.0, 1.0)
            active_rule_weight = float(rule_weight or 0.0)

        trig_threshold = float(trigger_threshold or DEFAULT_TRIGGER_THRESHOLD)
        rules_triggered = (rule_arr >= trig_threshold).astype(bool)

        if model_confidence is not None:
            conf_arr = np.asarray(model_confidence, dtype=np.float64).reshape(-1)
            if conf_arr.size == 0:
                conf_arr = np.zeros_like(model_arr)
            elif conf_arr.size == 1:
                conf_arr = np.full_like(model_arr, float(conf_arr[0]))
            elif conf_arr.size != model_arr.size:
                limit = min(conf_arr.size, model_arr.size)
                padded = np.empty_like(model_arr)
                padded[:limit] = conf_arr[:limit]
                if limit < model_arr.size:
                    padded[limit:] = conf_arr[limit - 1]
                conf_arr = padded
            conf_arr = np.clip(conf_arr, 0.0, 1.0)
        else:
            conf_arr = None

        model_w = float(model_weight or DEFAULT_MODEL_WEIGHT)
        rule_w = float(active_rule_weight)
        total = model_w + rule_w
        if not np.isfinite(total) or total <= 0.0:
            total = 1.0

        profile_key = str(profile or "baseline").strip().lower()
        if profile_key.startswith("agg"):
            high_pair = (0.5, 0.5)
            low_pair = (0.35, 0.65)
        else:
            high_pair = (0.8, 0.2)
            low_pair = (0.6, 0.4)

        if conf_arr is not None:
            high_mask = conf_arr >= 0.8
            model_weights = np.where(high_mask, total * high_pair[0], total * low_pair[0])
            rule_weights = np.where(high_mask, total * high_pair[1], total * low_pair[1])
            fused_scores = model_weights * model_arr + rule_weights * normalized_rules
        else:
            fused_scores = model_w * model_arr + rule_w * normalized_rules
        fused_flags = (fused_scores >= float(threshold or DEFAULT_FUSION_THRESHOLD)).astype(bool)

        return fused_scores, fused_flags, rules_triggered

try:  # Optional metrics helpers
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    accuracy_score = None  # type: ignore[assignment]
    precision_score = None  # type: ignore[assignment]
    recall_score = None  # type: ignore[assignment]
    f1_score = None  # type: ignore[assignment]
    confusion_matrix = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]

logger = get_logger(__name__)


METRIC_LABEL_CANDIDATES = ["LabelBinary", "Label", "label", "class", "ground_truth"]


def compute_detection_metrics(
    df,
    *,
    label_col: str = "Label",
    pred_col: str = "prediction_status",
    score_col: Optional[str] = None,
):
    """计算二分类检测的准确率、召回率、精度、F1、AUC 与混淆矩阵。"""

    if (
        accuracy_score is None
        or precision_score is None
        or recall_score is None
        or f1_score is None
        or confusion_matrix is None
        or roc_auc_score is None
        or pd is None
    ):
        return None

    if not hasattr(df, "columns") or label_col not in df.columns or pred_col not in df.columns:
        return None

    y_true = df[label_col]
    if y_true.dtype == "O":
        mapping = {
            "BENIGN": 0,
            "NORMAL": 0,
            "BENIGN/NEUTRAL": 0,
            "MALICIOUS": 1,
            "ATTACK": 1,
        }
        y_true = y_true.map(mapping)

    mask = y_true.isin([0, 1])
    if mask.sum() == 0:
        return None

    y_true = y_true[mask].astype(int)

    try:
        y_pred_numeric = pd.to_numeric(df.loc[mask, pred_col], errors="coerce")
    except Exception:
        return None

    pred_mask = y_pred_numeric.isin([0, 1])
    if pred_mask.sum() == 0:
        return None

    y_true = y_true[pred_mask]
    y_pred = y_pred_numeric[pred_mask].astype(int)

    score_series = None
    if score_col and score_col in df.columns:
        try:
            score_series = pd.to_numeric(df.loc[pred_mask, score_col], errors="coerce")
        except Exception:
            score_series = None

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0, pos_label=1)
    f1_val = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(acc),
        "precision": float(pre),
        "recall": float(rec),
        "f1": float(f1_val),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support": int(len(y_true)),
        "auc": None,
    }

    if score_series is not None and not score_series.isna().all():
        try:
            metrics["auc"] = float(roc_auc_score(y_true, score_series))
        except Exception:
            metrics["auc"] = None

    return metrics


def _safe_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _resolve_rule_config(metadata: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    metadata_obj = metadata if isinstance(metadata, dict) else {}

    profile_override = metadata_obj.get("rule_profile") if metadata_obj else None
    if not isinstance(profile_override, str) or not profile_override.strip():
        profile_override = None
    else:
        profile_override = profile_override.strip()

    try:
        config = get_rule_settings(profile=profile_override)
    except Exception:
        config = {}

    params = config.get("params") if isinstance(config, dict) else {}

    try:
        fusion_defaults = get_fusion_settings(profile=profile_override)
    except Exception:
        fusion_defaults = {
            "model_weight": DEFAULT_MODEL_WEIGHT,
            "rule_weight": DEFAULT_RULE_WEIGHT,
            "fusion_threshold": DEFAULT_FUSION_THRESHOLD,
            "profile": profile_override or "baseline",
        }

    profile = profile_override or fusion_defaults.get("profile") or (
        config.get("profile") if isinstance(config, dict) else None
    )
    profile_name = profile.strip() if isinstance(profile, str) and profile.strip() else None

    default_threshold = _safe_float(
        config.get("trigger_threshold") if isinstance(config, dict) else None,
        DEFAULT_TRIGGER_THRESHOLD,
    )
    default_model_weight = _safe_float(
        fusion_defaults.get("model_weight") if isinstance(fusion_defaults, dict) else None,
        DEFAULT_MODEL_WEIGHT,
    )
    default_rule_weight = _safe_float(
        fusion_defaults.get("rule_weight") if isinstance(fusion_defaults, dict) else None,
        DEFAULT_RULE_WEIGHT,
    )
    default_fusion_threshold = _safe_float(
        fusion_defaults.get("fusion_threshold") if isinstance(fusion_defaults, dict) else None,
        DEFAULT_FUSION_THRESHOLD,
    )

    rule_threshold = _safe_float(
        metadata_obj.get("rule_threshold") if metadata_obj else None,
        default_threshold,
    )
    fusion_threshold = _safe_float(
        metadata_obj.get("fusion_threshold") if metadata_obj else None,
        default_fusion_threshold,
    )

    model_weight = _safe_float(
        metadata_obj.get("fusion_model_weight") if metadata_obj else None,
        default_model_weight,
    )
    rule_weight = _safe_float(
        metadata_obj.get("fusion_rule_weight") if metadata_obj else None,
        default_rule_weight,
    )

    weights_meta = metadata_obj.get("fusion_weights") if metadata_obj else None
    if isinstance(weights_meta, dict) and weights_meta:
        normalized_model = _safe_float(weights_meta.get("model"), 1.0)
        normalized_rules = _safe_float(weights_meta.get("rules"), 0.0)
    else:
        total = float(model_weight + rule_weight)
        if not np.isfinite(total) or total <= 0.0:
            normalized_model = 1.0
            normalized_rules = 0.0
        else:
            normalized_model = float(model_weight) / total
            normalized_rules = float(rule_weight) / total

    return {
        "params": params if isinstance(params, dict) else {},
        "threshold": float(rule_threshold),
        "model_weight": float(model_weight),
        "rule_weight": float(rule_weight),
        "fusion_threshold": float(fusion_threshold),
        "profile": profile_name,
        "normalized_weights": {
            "model": float(normalized_model),
            "rules": float(normalized_rules),
        },
    }

def _run_prediction(
    pipeline_path: str,
    feature_csv: str,
    *,
    metadata_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, object]:
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"未找到模型管线: {pipeline_path}")
    if not os.path.exists(feature_csv):
        raise FileNotFoundError(f"未找到特征 CSV: {feature_csv}")

    metadata_payload: Dict[str, object] = {}
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                loaded_meta = json.load(fh)
            if isinstance(loaded_meta, dict):
                metadata_payload = loaded_meta
        except Exception as exc:  # pragma: no cover - metadata best effort
            logger.warning("Failed to load metadata %s: %s", metadata_path, exc)

    metadata_obj: Dict[str, object] = dict(metadata_payload)

    def _maybe_compute_metrics(frame) -> Optional[Dict[str, object]]:
        score_col = None
        for candidate_score in ("fusion_score", "malicious_score", "model_score"):
            if candidate_score in frame.columns:
                score_col = candidate_score
                break
        for candidate in METRIC_LABEL_CANDIDATES:
            metrics = compute_detection_metrics(
                frame,
                label_col=candidate,
                pred_col="prediction_status",
                score_col=score_col,
            )
            if metrics:
                metrics["label_column"] = candidate
                return metrics
        return None

    if joblib_load is None or pd is None or np is None:
        raise RuntimeError("缺少 numpy/pandas/joblib 依赖，无法加载完整模型。")

    pipeline = joblib_load(pipeline_path)
    try:
        df = read_csv_flexible(feature_csv)
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"无法读取特征 CSV，请检查文件编码：{exc}") from exc
    if df.empty:
        raise RuntimeError("特征 CSV 为空，无法预测。")

    if isinstance(pipeline, dict) and "model" in pipeline and "feature_names" in pipeline:
        feature_names = [str(name) for name in pipeline.get("feature_names", [])]
        if not feature_names:
            raise RuntimeError("模型缺少特征列描述，无法执行预测。")

        missing = [col for col in feature_names if col not in df.columns]
        if missing:
            sample = ", ".join(missing[:8])
            more = " ..." if len(missing) > 8 else ""
            raise RuntimeError(f"特征 CSV 缺少必要列: {sample}{more}")

        matrix = (
            df.loc[:, feature_names]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64, copy=False)
        )

        model = pipeline["model"]
        preds_arr = np.asarray(model.predict(matrix)).reshape(-1)

        label_mapping = pipeline.get("label_mapping")
        scores_arr = _extract_positive_probability(
            model,
            matrix,
            label_mapping if isinstance(label_mapping, dict) else None,
        )
        labels, _, _, _ = summarize_prediction_labels(
            preds_arr,
            label_mapping if isinstance(label_mapping, dict) else None,
        )

        model_flags_bool = np.zeros(preds_arr.shape[0], dtype=bool)
        model_statuses: List[str] = []
        for idx, label in enumerate(labels):
            if label == "异常":
                model_flags_bool[idx] = True
                model_statuses.append("异常")
            elif label == "正常":
                model_flags_bool[idx] = False
                model_statuses.append("正常")
            else:
                fallback_flag = bool(scores_arr[idx] >= 0.5)
                model_flags_bool[idx] = fallback_flag
                model_statuses.append("异常" if fallback_flag else "正常")

        pipeline_meta = pipeline.get("metadata") if isinstance(pipeline, dict) else None
        if isinstance(pipeline_meta, dict):
            metadata_obj.update(pipeline_meta)
        if metadata_payload:
            metadata_obj.update(metadata_payload)

        rule_settings = _resolve_rule_config(metadata_obj)
        rule_params = (
            rule_settings.get("params") if isinstance(rule_settings.get("params"), dict) else {}
        )
        rule_threshold_value = float(
            rule_settings.get("threshold", DEFAULT_TRIGGER_THRESHOLD)
        )
        rule_profile = (
            rule_settings.get("profile") if isinstance(rule_settings.get("profile"), str) else None
        )
        fusion_defaults = get_fusion_settings(profile=rule_profile)
        fusion_model_weight_base = float(
            rule_settings.get("model_weight", fusion_defaults["model_weight"])
        )
        fusion_rule_weight_base = float(
            rule_settings.get("rule_weight", fusion_defaults["rule_weight"])
        )
        fusion_threshold_value = float(
            rule_settings.get("fusion_threshold", fusion_defaults["fusion_threshold"])
        )
        rule_profile = (
            fusion_defaults.get("profile") if rule_profile is None else rule_profile
        )
        normalized_weights = rule_settings.get("normalized_weights")

        effective_rule_threshold = float(rule_threshold_value)

        rule_scores_array: Optional[np.ndarray] = None
        rule_reasons_list: Optional[List[str]] = None
        rule_flags_array: Optional[np.ndarray] = None
        if apply_risk_rules is not None and pd is not None:
            try:
                score_series, reason_series = apply_risk_rules(
                    df,
                    params=rule_params,
                    profile=rule_profile,
                )
            except Exception:
                score_series = None
                reason_series = None
            if score_series is not None:
                try:
                    rule_scores_array = score_series.astype(float, copy=False).to_numpy(dtype=float)
                except Exception:
                    rule_scores_array = np.asarray(score_series, dtype=np.float64).reshape(-1)
                if reason_series is not None:
                    rule_reasons_list = reason_series.astype(str, copy=False).tolist()
                rule_flags_array = rule_scores_array >= float(effective_rule_threshold)

        if rule_scores_array is not None and rule_scores_array.size == 0:
            rule_scores_array = None

        model_score_input = np.clip(scores_arr, 0.0, 1.0)
        model_confidence = np.maximum(model_score_input, 1.0 - model_score_input)

        fusion_scores_array, fusion_flags_array, rules_triggered_array = fuse_model_rule_votes(
            model_score_input,
            rule_scores_array,
            profile=rule_profile,
            model_weight=float(fusion_model_weight_base),
            rule_weight=float(fusion_rule_weight_base),
            threshold=float(fusion_threshold_value),
            trigger_threshold=float(effective_rule_threshold),
            model_confidence=model_confidence,
        )
        fusion_scores_array = np.asarray(fusion_scores_array, dtype=np.float64).reshape(-1)
        fusion_flags_array = np.asarray(fusion_flags_array, dtype=bool).reshape(-1)
        rules_triggered_array = np.asarray(rules_triggered_array, dtype=bool).reshape(-1)

        rule_flags_array = rules_triggered_array

        total_weight = float(fusion_model_weight_base + fusion_rule_weight_base)
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            fusion_weight_model = 1.0
            fusion_weight_rules = 0.0
        else:
            fusion_weight_model = float(fusion_model_weight_base) / total_weight
            fusion_weight_rules = float(fusion_rule_weight_base) / total_weight

        profile_key = (rule_profile or "").strip().lower()
        if rule_scores_array is not None:
            rule_scores_for_logic = np.asarray(rule_scores_array, dtype=np.float64).reshape(-1)
        else:
            rule_scores_for_logic = np.zeros_like(fusion_scores_array, dtype=np.float64)
        if rule_scores_for_logic.shape != fusion_scores_array.shape:
            target_len = fusion_scores_array.shape[0]
            current_len = rule_scores_for_logic.shape[0]
            if current_len < target_len:
                padded = np.zeros_like(fusion_scores_array, dtype=np.float64)
                padded[:current_len] = rule_scores_for_logic
                rule_scores_for_logic = padded
            else:
                rule_scores_for_logic = rule_scores_for_logic[:target_len]

        strong_rule_flags = np.logical_or(
            rule_scores_for_logic >= 80.0,
            rules_triggered_array,
        )
        if profile_key == "aggressive":
            final_flags_array = np.logical_or(
                fusion_scores_array >= float(fusion_threshold_value),
                strong_rule_flags,
            )
        else:
            final_flags_array = fusion_scores_array >= float(fusion_threshold_value)

        final_statuses = ["异常" if flag else "正常" for flag in final_flags_array]
        anomaly_count = int(np.count_nonzero(final_flags_array))
        normal_count = int(len(final_flags_array) - anomaly_count)
        status_text = "异常" if anomaly_count > 0 else ("正常" if len(final_flags_array) else None)

        output_df = df.copy()
        output_df["prediction"] = preds_arr
        output_df["prediction_label"] = labels
        output_df["malicious_score"] = scores_arr
        output_df["model_flag"] = model_flags_bool.astype(int)
        output_df["model_status"] = model_statuses
        output_df["fusion_score"] = fusion_scores_array
        output_df["fusion_decision"] = fusion_flags_array.astype(int)
        output_df["prediction_status"] = final_flags_array.astype(int)
        output_df["fusion_status"] = final_statuses
        if rule_scores_array is not None:
            output_df["rules_score"] = rule_scores_array
        else:
            output_df["rules_score"] = np.zeros(len(output_df), dtype=float)

        output_df["rules_flag"] = rule_flags_array.astype(int)
        output_df["rules_triggered"] = rule_flags_array.astype(bool)
        if rule_reasons_list is not None:
            output_df["rules_reasons"] = rule_reasons_list
        elif "rules_reasons" not in output_df.columns:
            output_df["rules_reasons"] = ["" for _ in range(len(output_df))]

        if output_path is None:
            base = Path(feature_csv).with_suffix("")
            output_path = str(base) + "_predictions.csv"
        output_df.to_csv(output_path, index=False, encoding="utf-8")

        result_info = {
            "output_path": output_path,
            "status_text": status_text,
            "anomaly_count": anomaly_count,
            "normal_count": normal_count,
            "fusion_threshold": float(fusion_threshold_value),
            "fusion_weights": {
                "model": fusion_weight_model,
                "rules": fusion_weight_rules,
            },
        }
        metrics = _maybe_compute_metrics(output_df)
        if metrics:
            result_info["metrics"] = metrics
        if rule_profile:
            result_info["rule_profile"] = rule_profile
    else:
        named_steps = getattr(pipeline, "named_steps", None)
        if isinstance(named_steps, dict) and "detector" in named_steps:
            raise RuntimeError("当前版本不再支持包含 detector 步骤的无监督异常检测管线。")

        raise RuntimeError(
            "不支持的管线格式，请使用包含 model 与 feature_names 的监督模型导出。"
        )

    if metadata_obj and "metadata" not in result_info:
        result_info["metadata"] = metadata_obj

    log_model_event(
        "cli.predict",
        {
            "pipeline_path": pipeline_path,
            "feature_csv": feature_csv,
            "output_path": result_info.get("output_path"),
            "status_text": result_info.get("status_text"),
            "anomaly_count": result_info.get("anomaly_count"),
        },
    )
    return result_info


def _resolve_positive_index(classes: Iterable[object], label_mapping: Optional[Dict[int, str]] = None) -> Optional[int]:
    candidates = list(classes) if classes is not None else []
    for idx, cls in enumerate(candidates):
        try:
            value = int(cls)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            value = None
        if value == 1:
            return idx
    for idx, cls in enumerate(candidates):
        try:
            value = int(cls)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if value == -1:
            return idx
        if label_mapping and value is not None:
            mapped = label_mapping.get(value)
            if isinstance(mapped, str) and mapped.upper().startswith("MAL"):
                return idx
    return None


def _extract_positive_probability(
    model: object,
    matrix: "np.ndarray",
    label_mapping: Optional[Dict[int, str]] = None,
) -> "np.ndarray":
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(matrix)
        except Exception:
            proba = None
        if proba is not None:
            proba_arr = np.asarray(proba, dtype=np.float64)
            if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                classes = getattr(model, "classes_", None)
                pos_index = _resolve_positive_index(classes, label_mapping)
                if pos_index is None:
                    pos_index = proba_arr.shape[1] - 1
                pos_index = max(0, min(proba_arr.shape[1] - 1, int(pos_index)))
                return proba_arr[:, pos_index]
            return proba_arr.reshape(-1)

    if hasattr(model, "decision_function"):
        try:
            decision = model.decision_function(matrix)
        except Exception:
            decision = None
        if decision is not None:
            decision_arr = np.asarray(decision, dtype=np.float64)
            if decision_arr.ndim == 1:
                return 1.0 / (1.0 + np.exp(-decision_arr))
            if decision_arr.ndim == 2 and decision_arr.shape[1] >= 2:
                classes = getattr(model, "classes_", None)
                pos_index = _resolve_positive_index(classes, label_mapping)
                if pos_index is None:
                    pos_index = decision_arr.shape[1] - 1
                pos_index = max(0, min(decision_arr.shape[1] - 1, int(pos_index)))
                return decision_arr[:, pos_index]
            return decision_arr.reshape(-1)

    try:
        predictions = model.predict(matrix)
    except Exception:
        predictions = np.zeros(matrix.shape[0])
    return np.asarray(predictions, dtype=np.float64).reshape(-1)


def _handle_extract(args: argparse.Namespace) -> int:
    if extract_features_dir is None:
        raise RuntimeError("提取功能需要 dpkt 等可选依赖。")
    output = extract_features_dir(
        args.pcap_dir,
        args.output_dir,
        workers=args.workers,
        fast=args.fast,
    )
    for path in output:
        print(path)
    log_model_event(
        "cli.extract",
        {"pcap_dir": args.pcap_dir, "output_dir": args.output_dir, "files": len(output)},
    )
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    if ModelTrainer is None:
        raise RuntimeError("缺少建模依赖（如 scikit-learn），无法执行训练流程。")

    trainer = ModelTrainer()
    train_kwargs: Dict[str, object] = {}
    if getattr(args, "max_ensemble_members", None) is not None:
        train_kwargs["max_ensemble_members"] = args.max_ensemble_members
    if getattr(args, "ensemble_weight_metric", None):
        train_kwargs["ensemble_weight_metric"] = args.ensemble_weight_metric
    if getattr(args, "reset_ensemble", False):
        train_kwargs["reset_ensemble"] = True
    result = trainer.train_from_split(
        args.split_dir,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        **train_kwargs,
    )
    output_path = result.get("pipeline_latest") if isinstance(result, dict) else result
    summary = {
        "results_csv": result.get("results_csv") if isinstance(result, dict) else None,
        "model_path": output_path,
        "metadata_path": result.get("metadata_path") if isinstance(result, dict) else None,
        "packets": result.get("packets") if isinstance(result, dict) else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if output_path:
        print(output_path)
    log_model_event(
        "cli.train",
        {
            "split_dir": args.split_dir,
            "results_dir": args.results_dir,
            "models_dir": args.models_dir,
            "max_ensemble_members": getattr(args, "max_ensemble_members", None),
            "ensemble_weight_metric": getattr(args, "ensemble_weight_metric", None),
            "reset_ensemble": bool(getattr(args, "reset_ensemble", False)),
        },
    )
    return 0


def _handle_predict(args: argparse.Namespace) -> int:
    result = _run_prediction(
        args.pipeline,
        args.features,
        metadata_path=args.metadata,
        output_path=args.output,
    )
    status_text = result.get("status_text") if isinstance(result, dict) else None
    if status_text:
        anomaly = result.get("anomaly_count") if isinstance(result, dict) else None
        normal = result.get("normal_count") if isinstance(result, dict) else None
        if isinstance(anomaly, int) and isinstance(normal, int):
            print(f"预测结果：{status_text}（异常 {anomaly} / 正常 {normal}）")
        else:
            print(f"预测结果：{status_text}")
    output_path = result.get("output_path") if isinstance(result, dict) else result
    print(output_path)
    return 0


def _handle_analyze(args: argparse.Namespace) -> int:
    if analyze_results is None:
        raise RuntimeError("分析功能需要额外依赖 (matplotlib/pandas)。")
    result = analyze_results(
        args.results_csv,
        args.output_dir,
        metadata_path=args.metadata,
    )
    summary_path = result.get("summary_json") if isinstance(result, dict) else None
    if summary_path:
        print(summary_path)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else str(result))
    log_model_event(
        "cli.analyze",
        {"results_csv": args.results_csv, "output_dir": args.output_dir},
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maldet-service", description="Pipeline service CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="批量提取 PCAP 目录特征")
    p_extract.add_argument("pcap_dir", help="含有 PCAP/PCAPNG 的目录")
    p_extract.add_argument("output_dir", help="特征 CSV 输出目录")
    p_extract.add_argument("--workers", type=int, default=4, help="并发线程数")
    p_extract.add_argument("--fast", action="store_true", help="使用快速模式（可能精度稍低）")
    p_extract.set_defaults(func=_handle_extract)

    p_train = sub.add_parser("train", help="训练/追加监督集成模型")
    p_train.add_argument("split_dir", help="预处理数据集或 PCAP 目录")
    p_train.add_argument("results_dir", help="训练结果目录")
    p_train.add_argument("models_dir", help="模型输出目录")
    p_train.add_argument(
        "--max-ensemble-members",
        type=int,
        help="保留的最大集成模型数量（包含最新模型）",
    )
    p_train.add_argument(
        "--ensemble-weight-metric",
        help="集成模型权重依据（samples、auc、f1）",
    )
    p_train.add_argument(
        "--reset-ensemble",
        action="store_true",
        help="完全重训：清空历史模型，仅保留本次训练",
    )
    p_train.set_defaults(func=_handle_train)

    p_predict = sub.add_parser("predict", help="使用训练好的管线进行预测")
    p_predict.add_argument("pipeline", help="Pipeline joblib 路径")
    p_predict.add_argument("features", help="特征 CSV 路径")
    p_predict.add_argument("--metadata", help="模型元数据 JSON 路径")
    p_predict.add_argument("--output", help="预测结果输出 CSV")
    p_predict.set_defaults(func=_handle_predict)

    p_analyze = sub.add_parser("analyze", help="分析模型预测结果")
    p_analyze.add_argument("results_csv", help="预测结果 CSV")
    p_analyze.add_argument("output_dir", help="分析输出目录")
    p_analyze.add_argument("--metadata", help="模型元数据路径")
    p_analyze.set_defaults(func=_handle_analyze)

    return parser


def run_cli(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


# --- Optional FastAPI application -----------------------------------------

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional import guard
    FastAPI = None
    BaseModel = object  # type: ignore


if FastAPI is not None:

    class TrainRequest(BaseModel):
        split_dir: str
        results_dir: str
        models_dir: str
        max_ensemble_members: Optional[int] = None
        ensemble_weight_metric: Optional[str] = None
        reset_ensemble: Optional[bool] = False

    class PredictRequest(BaseModel):
        pipeline_path: str
        feature_csv: str
        metadata_path: Optional[str] = None
        output_path: Optional[str] = None

    class AnalyzeRequest(BaseModel):
        results_csv: str
        output_dir: str
        metadata_path: Optional[str] = None


    def create_app() -> FastAPI:
        app = FastAPI(title="MalDet Pipeline Service")

        @app.post("/train")
        def train_endpoint(req: TrainRequest):
            try:
                if ModelTrainer is None:
                    raise RuntimeError("缺少建模依赖（如 scikit-learn），无法执行训练流程。")
                trainer = ModelTrainer()
                train_kwargs: Dict[str, object] = {}
                if req.max_ensemble_members is not None:
                    train_kwargs["max_ensemble_members"] = req.max_ensemble_members
                if req.ensemble_weight_metric:
                    train_kwargs["ensemble_weight_metric"] = req.ensemble_weight_metric
                if req.reset_ensemble:
                    train_kwargs["reset_ensemble"] = True
                result = trainer.train_from_split(
                    req.split_dir,
                    results_dir=req.results_dir,
                    models_dir=req.models_dir,
                    **train_kwargs,
                )
                log_model_event(
                    "rest.train",
                    {
                        "split_dir": req.split_dir,
                        "results_dir": req.results_dir,
                        "models_dir": req.models_dir,
                        "max_ensemble_members": req.max_ensemble_members,
                        "ensemble_weight_metric": req.ensemble_weight_metric,
                        "reset_ensemble": bool(req.reset_ensemble),
                    },
                )
                return result
            except Exception as exc:  # pragma: no cover - runtime error surface
                logger.exception("REST train failed")
                raise HTTPException(status_code=500, detail=str(exc))

        @app.post("/predict")
        def predict_endpoint(req: PredictRequest):
            try:
                output = _run_prediction(
                    req.pipeline_path,
                    req.feature_csv,
                    metadata_path=req.metadata_path,
                    output_path=req.output_path,
                )
                return {"output": output}
            except Exception as exc:  # pragma: no cover - runtime error surface
                logger.exception("REST predict failed")
                raise HTTPException(status_code=500, detail=str(exc))

        @app.post("/analyze")
        def analyze_endpoint(req: AnalyzeRequest):
            try:
                result = analyze_results(
                    req.results_csv,
                    req.output_dir,
                    metadata_path=req.metadata_path,
                )
                log_model_event(
                    "rest.analyze",
                    {
                        "results_csv": req.results_csv,
                        "output_dir": req.output_dir,
                    },
                )
                return result
            except Exception as exc:  # pragma: no cover
                logger.exception("REST analyze failed")
                raise HTTPException(status_code=500, detail=str(exc))

        return app

else:  # pragma: no cover - FastAPI not available

    def create_app():  # type: ignore
        raise RuntimeError("FastAPI 未安装，无法创建 REST 服务。")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(run_cli())
