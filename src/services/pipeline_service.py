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
    from src.functions.modeling import (
        compute_risk_components,
        summarize_prediction_labels,
        train_unsupervised_on_split,
    )
except ModuleNotFoundError:  # pragma: no cover - triggered in lightweight envs
    compute_risk_components = None  # type: ignore[assignment]
    train_unsupervised_on_split = None  # type: ignore[assignment]

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

try:  # Optional lightweight inference helpers.
    from src.functions.simple_unsupervised import (
        simple_predict,
        load_simple_model,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    simple_predict = None  # type: ignore[assignment]
    load_simple_model = None  # type: ignore[assignment]

try:
    from src.functions.risk_rules import (  # type: ignore
        fuse_model_rule_votes,
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

    def fuse_model_rule_votes(  # type: ignore
        model_flags: Iterable[object],
        rule_scores: Optional[Iterable[object]],
        *,
        model_weight: float = DEFAULT_MODEL_WEIGHT,
        rule_weight: float = DEFAULT_RULE_WEIGHT,
        threshold: float = DEFAULT_FUSION_THRESHOLD,
    ) -> Tuple[np.ndarray, np.ndarray]:
        model_arr = np.asarray(model_flags, dtype=np.float64).reshape(-1)
        if model_arr.size == 0:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty.astype(bool)

        model_arr = np.clip(model_arr, 0.0, 1.0)

        if rule_scores is None:
            normalized_rules = np.zeros_like(model_arr)
            active_rule_weight = 0.0
        else:
            try:
                rule_arr = np.asarray(rule_scores, dtype=np.float64).reshape(-1)
            except Exception:
                rule_arr = None
            if rule_arr is None or rule_arr.size == 0:
                normalized_rules = np.zeros_like(model_arr)
                active_rule_weight = 0.0
            else:
                if rule_arr.size != model_arr.size:
                    if rule_arr.size < model_arr.size:
                        padded = np.zeros_like(model_arr)
                        padded[: rule_arr.size] = rule_arr
                        rule_arr = padded
                    else:
                        rule_arr = rule_arr[: model_arr.size]
                normalized_rules = np.clip(rule_arr / 100.0, 0.0, 1.0)
                active_rule_weight = float(rule_weight)

        total_weight = float(model_weight + active_rule_weight)
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            model_w = 1.0
            rule_w = 0.0
        else:
            model_w = float(model_weight) / total_weight
            rule_w = float(active_rule_weight) / total_weight

        fused_scores = model_w * model_arr + rule_w * normalized_rules
        fused_flags = fused_scores >= float(threshold)

        return fused_scores, fused_flags

logger = get_logger(__name__)


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
    profile = (
        profile_override
        if profile_override
        else (config.get("profile") if isinstance(config, dict) else None)
    )
    profile_name = profile.strip() if isinstance(profile, str) and profile.strip() else None

    default_threshold = _safe_float(
        config.get("trigger_threshold") if isinstance(config, dict) else None,
        DEFAULT_TRIGGER_THRESHOLD,
    )
    default_model_weight = _safe_float(
        config.get("model_weight") if isinstance(config, dict) else None,
        DEFAULT_MODEL_WEIGHT,
    )
    default_rule_weight = _safe_float(
        config.get("rule_weight") if isinstance(config, dict) else None,
        DEFAULT_RULE_WEIGHT,
    )
    default_fusion_threshold = _safe_float(
        config.get("fusion_threshold") if isinstance(config, dict) else None,
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

    if pipeline_path.lower().endswith(".json"):
        if load_simple_model is None or simple_predict is None:
            raise RuntimeError(
                "当前环境不支持 JSON 模型推理，请安装 simple_unsupervised 模块或提供完整模型管线。"
            )
        model = load_simple_model(pipeline_path)
        output_path, _ = simple_predict(
            model,
            feature_csv,
            output_path=output_path,
        )
        result_info: Dict[str, object] = {
            "output_path": output_path,
            "status_text": None,
            "anomaly_count": None,
            "normal_count": None,
        }
    else:
        if compute_risk_components is None or train_unsupervised_on_split is None:
            raise RuntimeError(
                "缺少建模依赖（如 scikit-learn），无法加载完整模型。"
            )
        if joblib_load is None or pd is None or np is None:
            raise RuntimeError(
                "缺少 numpy/pandas/joblib 依赖，无法加载完整模型。"
            )

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

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(matrix)
                if np.ndim(proba) == 2:
                    scores_arr = np.asarray(proba, dtype=np.float64).max(axis=1)
                else:
                    scores_arr = np.asarray(proba, dtype=np.float64).reshape(-1)
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(matrix)
                decision_arr = np.asarray(decision, dtype=np.float64)
                if decision_arr.ndim == 1:
                    scores_arr = 1.0 / (1.0 + np.exp(-decision_arr))
                else:
                    scores_arr = decision_arr.max(axis=1)
            else:
                scores_arr = np.asarray(preds_arr, dtype=np.float64)

            label_mapping = pipeline.get("label_mapping")
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
            rule_params = rule_settings.get("params") if isinstance(rule_settings.get("params"), dict) else {}
            rule_threshold_value = float(
                rule_settings.get("threshold", DEFAULT_TRIGGER_THRESHOLD)
            )
            fusion_model_weight_base = float(rule_settings.get("model_weight", DEFAULT_MODEL_WEIGHT))
            fusion_rule_weight_base = float(rule_settings.get("rule_weight", DEFAULT_RULE_WEIGHT))
            fusion_threshold_value = float(rule_settings.get("fusion_threshold", DEFAULT_FUSION_THRESHOLD))
            rule_profile = rule_settings.get("profile") if isinstance(rule_settings.get("profile"), str) else None
            normalized_weights = rule_settings.get("normalized_weights")

            effective_rule_threshold = min(
                float(rule_threshold_value),
                float(RULE_TRIGGER_THRESHOLD),
            )

            rule_scores_array: Optional[np.ndarray] = None
            rule_reasons_list: Optional[List[str]] = None
            rule_flags_array: Optional[np.ndarray] = None
            if apply_risk_rules is not None and pd is not None:
                try:
                    score_series, reason_series = apply_risk_rules(df, params=rule_params)
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

            active_rule_weight = (
                float(fusion_rule_weight_base)
                if rule_scores_array is not None and rule_scores_array.size > 0
                else 0.0
            )

            fusion_scores_array, fusion_flags_array = fuse_model_rule_votes(
                model_flags_bool.astype(np.float64),
                rule_scores_array,
                model_weight=float(fusion_model_weight_base),
                rule_weight=active_rule_weight,
                threshold=float(fusion_threshold_value),
            )
            fusion_scores_array = np.asarray(fusion_scores_array, dtype=np.float64).reshape(-1)
            fusion_flags_array = np.asarray(fusion_flags_array, dtype=bool).reshape(-1)

            if rule_flags_array is not None and rule_flags_array.size:
                rule_bool = np.asarray(rule_flags_array, dtype=bool).reshape(-1)
                if rule_bool.size < fusion_flags_array.size:
                    rule_bool = np.pad(
                        rule_bool,
                        (0, fusion_flags_array.size - rule_bool.size),
                        constant_values=False,
                    )
                elif rule_bool.size > fusion_flags_array.size:
                    rule_bool = rule_bool[: fusion_flags_array.size]
                fusion_flags_array = np.logical_or(fusion_flags_array, rule_bool)
                threshold_floor = float(fusion_threshold_value)
                if np.isfinite(threshold_floor):
                    fusion_scores_array = np.where(
                        rule_bool,
                        np.maximum(fusion_scores_array, threshold_floor),
                        fusion_scores_array,
                    )

            total_weight = float(fusion_model_weight_base + active_rule_weight)
            if not np.isfinite(total_weight) or total_weight <= 0.0:
                fusion_weight_model = 1.0
                fusion_weight_rules = 0.0
            else:
                fusion_weight_model = float(fusion_model_weight_base) / total_weight
                fusion_weight_rules = float(active_rule_weight) / total_weight

            final_statuses = ["异常" if flag else "正常" for flag in fusion_flags_array]
            anomaly_count = int(fusion_flags_array.sum())
            normal_count = int(len(fusion_flags_array) - anomaly_count)
            status_text = "异常" if anomaly_count > 0 else ("正常" if len(fusion_flags_array) else None)

            output_df = df.copy()
            output_df["prediction"] = preds_arr
            output_df["prediction_label"] = labels
            output_df["malicious_score"] = scores_arr
            output_df["model_flag"] = model_flags_bool.astype(int)
            output_df["model_status"] = model_statuses
            output_df["fusion_score"] = fusion_scores_array
            output_df["fusion_decision"] = fusion_flags_array.astype(int)
            output_df["prediction_status"] = final_statuses
            if rule_scores_array is not None:
                output_df["rules_score"] = rule_scores_array
                if rule_flags_array is not None:
                    output_df["rules_flag"] = rule_flags_array.astype(int)
                    output_df["rules_triggered"] = rule_flags_array.astype(bool)
                else:
                    fallback_flags = rule_scores_array >= float(effective_rule_threshold)
                    output_df["rules_flag"] = fallback_flags.astype(int)
                    output_df["rules_triggered"] = fallback_flags.astype(bool)
                if rule_reasons_list is not None:
                    output_df["rules_reasons"] = rule_reasons_list

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
            if rule_profile:
                result_info["rule_profile"] = rule_profile
        else:
            metadata: Dict[str, object] = dict(metadata_payload)

            detector = pipeline.named_steps.get("detector")
            if detector is None:
                raise RuntimeError("管线中缺少 detector 步骤。")

            transformed = pipeline[:-1].transform(df)
            scores = detector.score_samples(transformed)
            preds = detector.predict(transformed)

            vote_ratio = detector.last_vote_ratio_
            if vote_ratio is None and detector.fit_votes_:
                vote_ratio = np.vstack(
                    [np.where(v == -1, 1.0, 0.0) for v in detector.fit_votes_.values()]
                ).mean(axis=0)
            if vote_ratio is None:
                vote_ratio = np.ones_like(scores)

            threshold = metadata.get("threshold") if isinstance(metadata, dict) else None
            if threshold is None and getattr(detector, "threshold_", None) is not None:
                threshold = float(detector.threshold_)
            elif threshold is None:
                threshold = float(np.quantile(scores, 0.05))

            score_std = metadata.get("score_std") if isinstance(metadata, dict) else None
            if score_std is None:
                score_std = float(np.std(scores) or 1.0)

            vote_threshold = metadata.get("vote_threshold") if isinstance(metadata, dict) else None
            if vote_threshold is None and getattr(detector, "vote_threshold_", None) is not None:
                vote_threshold = float(detector.vote_threshold_)
            elif vote_threshold is None:
                vote_threshold = float(np.clip(np.mean(vote_ratio), 0.0, 1.0))

            risk_score, score_component, vote_component = compute_risk_components(
                scores,
                vote_ratio,
                float(threshold),
                float(vote_threshold),
                float(score_std),
            )

            output_df = df.copy()
            output_df["anomaly_score"] = scores
            output_df["vote_ratio"] = vote_ratio
            output_df["score_component"] = score_component
            output_df["vote_component"] = vote_component
            output_df["risk_score"] = risk_score
            output_df["prediction"] = preds

            model_flags_bool = (preds == -1).astype(bool)
            model_statuses = ["异常" if flag else "正常" for flag in model_flags_bool]

            rule_settings = _resolve_rule_config(metadata)
            rule_params = rule_settings.get("params") if isinstance(rule_settings.get("params"), dict) else {}
            rule_threshold_value = float(
                rule_settings.get("threshold", DEFAULT_TRIGGER_THRESHOLD)
            )
            fusion_model_weight_base = float(rule_settings.get("model_weight", DEFAULT_MODEL_WEIGHT))
            fusion_rule_weight_base = float(rule_settings.get("rule_weight", DEFAULT_RULE_WEIGHT))
            fusion_threshold_value = float(rule_settings.get("fusion_threshold", DEFAULT_FUSION_THRESHOLD))
            rule_profile = rule_settings.get("profile") if isinstance(rule_settings.get("profile"), str) else None
            normalized_weights = rule_settings.get("normalized_weights")

            effective_rule_threshold = min(
                float(rule_threshold_value),
                float(RULE_TRIGGER_THRESHOLD),
            )

            rule_scores_array: Optional[np.ndarray] = None
            rule_reasons_list: Optional[List[str]] = None
            rule_flags_array: Optional[np.ndarray] = None
            if apply_risk_rules is not None and pd is not None:
                try:
                    score_series, reason_series = apply_risk_rules(df, params=rule_params)
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

            active_rule_weight = (
                float(fusion_rule_weight_base)
                if rule_scores_array is not None and rule_scores_array.size > 0
                else 0.0
            )

            fusion_scores_array, fusion_flags_array = fuse_model_rule_votes(
                model_flags_bool.astype(np.float64),
                rule_scores_array,
                model_weight=float(fusion_model_weight_base),
                rule_weight=active_rule_weight,
                threshold=float(fusion_threshold_value),
            )
            fusion_scores_array = np.asarray(fusion_scores_array, dtype=np.float64).reshape(-1)
            fusion_flags_array = np.asarray(fusion_flags_array, dtype=bool).reshape(-1)

            if rule_flags_array is not None and rule_flags_array.size:
                rule_bool = np.asarray(rule_flags_array, dtype=bool).reshape(-1)
                if rule_bool.size < fusion_flags_array.size:
                    rule_bool = np.pad(
                        rule_bool,
                        (0, fusion_flags_array.size - rule_bool.size),
                        constant_values=False,
                    )
                elif rule_bool.size > fusion_flags_array.size:
                    rule_bool = rule_bool[: fusion_flags_array.size]
                fusion_flags_array = np.logical_or(fusion_flags_array, rule_bool)
                threshold_floor = float(fusion_threshold_value)
                if np.isfinite(threshold_floor):
                    fusion_scores_array = np.where(
                        rule_bool,
                        np.maximum(fusion_scores_array, threshold_floor),
                        fusion_scores_array,
                    )

            total_weight = float(fusion_model_weight_base + active_rule_weight)
            if not np.isfinite(total_weight) or total_weight <= 0.0:
                fusion_weight_model = 1.0
                fusion_weight_rules = 0.0
            else:
                fusion_weight_model = float(fusion_model_weight_base) / total_weight
                fusion_weight_rules = float(active_rule_weight) / total_weight

            final_statuses = ["异常" if flag else "正常" for flag in fusion_flags_array]
            anomaly_count = int(fusion_flags_array.sum())
            normal_count = int(len(fusion_flags_array) - anomaly_count)
            status_text = "异常" if anomaly_count > 0 else ("正常" if len(fusion_flags_array) else None)

            output_df["model_flag"] = model_flags_bool.astype(int)
            output_df["model_status"] = model_statuses
            output_df["fusion_score"] = fusion_scores_array
            output_df["fusion_decision"] = fusion_flags_array.astype(int)
            output_df["prediction_status"] = final_statuses
            output_df["is_malicious"] = fusion_flags_array.astype(int)

            if rule_scores_array is not None:
                output_df["rules_score"] = rule_scores_array
                if rule_flags_array is not None:
                    output_df["rules_flag"] = rule_flags_array.astype(int)
                    output_df["rules_triggered"] = rule_flags_array.astype(bool)
                else:
                    fallback_flags = rule_scores_array >= float(effective_rule_threshold)
                    output_df["rules_flag"] = fallback_flags.astype(int)
                    output_df["rules_triggered"] = fallback_flags.astype(bool)
                if rule_reasons_list is not None:
                    output_df["rules_reasons"] = rule_reasons_list

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
            if rule_profile:
                result_info["rule_profile"] = rule_profile
            if isinstance(normalized_weights, dict):
                result_info["configured_fusion_weights"] = {
                    "model": float(normalized_weights.get("model", fusion_weight_model)),
                    "rules": float(normalized_weights.get("rules", fusion_weight_rules)),
                }
            elif math.isfinite(fusion_model_weight_base + fusion_rule_weight_base) and (
                fusion_model_weight_base + fusion_rule_weight_base
            ) > 0.0:
                total_config = float(fusion_model_weight_base + fusion_rule_weight_base)
                result_info["configured_fusion_weights"] = {
                    "model": float(fusion_model_weight_base) / total_config,
                    "rules": float(fusion_rule_weight_base) / total_config,
                }
            else:
                result_info["configured_fusion_weights"] = {
                    "model": fusion_weight_model,
                    "rules": fusion_weight_rules,
                }
            if metadata_obj:
                result_info["metadata"] = metadata_obj

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
    if train_unsupervised_on_split is None:
        raise RuntimeError("缺少建模依赖（如 scikit-learn），无法执行训练流程。")
    train_kwargs: Dict[str, object] = {}
    if getattr(args, "max_ensemble_members", None) is not None:
        train_kwargs["max_ensemble_members"] = args.max_ensemble_members
    if getattr(args, "ensemble_weight_metric", None):
        train_kwargs["ensemble_weight_metric"] = args.ensemble_weight_metric
    if getattr(args, "reset_ensemble", False):
        train_kwargs["reset_ensemble"] = True
    result = train_unsupervised_on_split(
        args.split_dir,
        args.results_dir,
        args.models_dir,
        **train_kwargs,
    )
    summary = {
        "results_csv": result.get("results_csv"),
        "model_path": result.get("pipeline_path"),
        "metadata_path": result.get("metadata_path"),
        "packets": result.get("packets"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
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

    p_train = sub.add_parser("train", help="训练无监督模型")
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
                train_kwargs: Dict[str, object] = {}
                if req.max_ensemble_members is not None:
                    train_kwargs["max_ensemble_members"] = req.max_ensemble_members
                if req.ensemble_weight_metric:
                    train_kwargs["ensemble_weight_metric"] = req.ensemble_weight_metric
                if req.reset_ensemble:
                    train_kwargs["reset_ensemble"] = True
                result = train_unsupervised_on_split(
                    req.split_dir,
                    req.results_dir,
                    req.models_dir,
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