"""无监督异常检测训练流程。"""

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from joblib import dump

from src.functions.feature_extractor import extract_features, extract_features_dir
from src.functions.anomaly_detector import EnsembleAnomalyDetector
from src.functions.logging_utils import get_logger
from src.functions.transformers import (
    FeatureAligner,
    FeatureWeighter,
    PreprocessPipeline,
    DeepFeatureExtractor,
)

logger = get_logger(__name__)

MODEL_SCHEMA_VERSION = "2025.10"

META_COLUMNS = {
    "pcap_file",
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "__source_file__",
    "__source_path__",
}

GROUND_TRUTH_COLUMN_CANDIDATES = (
    "label",
    "labels",
    "ground_truth",
    "attack",
    "attacks",
    "is_attack",
    "is_malicious",
    "malicious",
    "malware",
    "anomaly",
)

NORMAL_TOKENS = {
    "normal",
    "normal.",
    "benign",
    "benign.",
    "good",
    "legit",
    "legitimate",
    "clean",
    "ok",
    "allow",
    "allowed",
    "pass",
    "passed",
    "success",
    "success.",
    "successful",
    "合法",
    "良性",
    "正常",
    "正常.",
    "正常流量",
    "正常登录",
    "成功",
    "成功登录",
    "登录成功",
    "登陆成功",
    "成功登陆",
    "成功登入",
    "登入成功",
    "4399",
    "4399登录",
    "4399成功登录",
    "baidu",
    "baidu.",
    "百度",
    "百度.",
    "百度登录",
    "百度成功登录",
    "0",
    "false",
    "no",
}

ANOMALY_TOKENS = {
    "attack",
    "attack.",
    "attacks",
    "intrusion",
    "intrusion.",
    "anomaly",
    "anomaly.",
    "abnormal",
    "malicious",
    "malicious.",
    "malware",
    "botnet",
    "spam",
    "ddos",
    "dos",
    "denied",
    "blocked",
    "fail",
    "failed",
    "failure",
    "error",
    "失败",
    "登录失败",
    "登陆失败",
    "登入失败",
    "非法",
    "恶意",
    "恶意流量",
    "攻击",
    "攻击流量",
    "异常",
    "异常流量",
    "异常登录",
    "可疑",
    "可疑流量",
    "嫌疑",
    "嫌疑流量",
    "1",
    "true",
    "yes",
}


def _build_token_variants(tokens: set[str]) -> set[str]:
    variants: set[str] = set()
    strip_chars = "。.,，!！?？；;:"
    for token in tokens:
        if not token:
            continue
        base = token.strip().lower()
        collapsed = base.replace(" ", "")
        trimmed = base.strip(strip_chars)
        collapsed_trimmed = collapsed.strip(strip_chars)
        for item in (base, collapsed, trimmed, collapsed_trimmed):
            if item:
                variants.add(item)
    return variants


NORMAL_TOKEN_VARIANTS = _build_token_variants(NORMAL_TOKENS)
ANOMALY_TOKEN_VARIANTS = _build_token_variants(ANOMALY_TOKENS)


def _is_npz(path: str) -> bool:
    return path.lower().endswith(".npz")


def _is_npy(path: str) -> bool:
    return path.lower().endswith(".npy")


def _latest_npz_in_dir(directory: str) -> Optional[str]:
    candidates = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if _is_npz(name)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def _is_preprocessed_csv(path: str) -> bool:
    if not path.lower().endswith(".csv"):
        return False
    name = os.path.basename(path)
    if not name.startswith("dataset_preprocessed_"):
        return False
    base, _ = os.path.splitext(path)
    return os.path.exists(f"{base}_meta.json")


def _series_to_binary(series: pd.Series) -> Optional[np.ndarray]:
    """尝试把标签列解析为 0/1，并允许未标注值。"""

    if series.empty:
        return None

    if series.dtype == bool:
        arr = series.fillna(False).astype(int).to_numpy()
        return arr.astype(float) if arr.size else None

    try:
        numeric = pd.to_numeric(series, errors="coerce")
    except Exception:
        numeric = None

    if numeric is not None and not numeric.isna().all():
        arr = numeric.to_numpy(dtype=float)
        binary = np.full(arr.shape, np.nan, dtype=float)
        finite_mask = np.isfinite(arr)
        if finite_mask.any():
            positive_mask = finite_mask & (arr > 0)
            zero_mask = finite_mask & (arr == 0)
            binary[positive_mask] = 1.0
            binary[zero_mask] = 0.0
        if np.isfinite(binary).any():
            return binary

    normalized = (
        series.fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\t\n\r\u3000]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )

    if normalized.eq("unknown").all():
        return None

    parsed = np.full(len(normalized), np.nan, dtype=float)

    for idx, value in enumerate(normalized):
        candidates = {
            value,
            value.replace(" ", ""),
            value.strip("。.,，!！?？；;:"),
            value.replace(" ", "").strip("。.,，!！?？；;:"),
        }
        if any(candidate in NORMAL_TOKEN_VARIANTS for candidate in candidates if candidate):
            parsed[idx] = 0.0
            continue
        if any(candidate in ANOMALY_TOKEN_VARIANTS for candidate in candidates if candidate):
            parsed[idx] = 1.0
            continue
        try:
            numeric_value = float(value)
        except Exception:
            numeric_value = None
        if numeric_value is None:
            continue
        if numeric_value > 0:
            parsed[idx] = 1.0
        elif numeric_value == 0:
            parsed[idx] = 0.0

    labeled_mask = np.isfinite(parsed)
    if not labeled_mask.any():
        return None

    return parsed


def _extract_ground_truth(df: pd.DataFrame) -> Tuple[Optional[str], Optional[np.ndarray]]:
    for column in df.columns:
        if column.lower() in GROUND_TRUTH_COLUMN_CANDIDATES:
            series = df[column]
            binary = _series_to_binary(series)
            if binary is not None and binary.size == len(df):
                return column, binary
    return None, None


def _latest_preprocessed_dataset(directory: str) -> Optional[str]:
    candidates: List[str] = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if _is_preprocessed_csv(path) or _is_npz(path) or _is_npy(path):
            base = os.path.basename(path)
            if not base.startswith("dataset_preprocessed_"):
                continue
            base_root, _ = os.path.splitext(path)
            if os.path.exists(f"{base_root}_meta.json"):
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def _manifest_path_for(dataset_path: str) -> str:
    base, _ = os.path.splitext(dataset_path)
    return f"{base}_manifest.csv"


def _meta_path_for(dataset_path: str) -> str:
    base, _ = os.path.splitext(dataset_path)
    return f"{base}_meta.json"


def _resolve_dataset_from_auxiliary(path: str) -> str:
    """当用户误选 manifest/meta 辅助文件时，回退到对应的数据集文件。"""

    if not path:
        return path

    lower = path.lower()

    def _candidate(base: str) -> Optional[str]:
        for ext in (".npy", ".npz", ".csv"):
            candidate = f"{base}{ext}"
            if os.path.exists(candidate):
                return candidate
        return None

    if lower.endswith("_manifest.csv"):
        base = path[: -len("_manifest.csv")]
        resolved = _candidate(base)
        if resolved:
            return resolved

    if lower.endswith("_meta.json"):
        base = path[: -len("_meta.json")]
        resolved = _candidate(base)
        if resolved:
            return resolved

    return path


def _load_npz_dataset(dataset_path: str) -> Tuple[np.ndarray, list]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    if "X" not in data:
        raise ValueError(f"NPZ 文件缺少 'X' 数据: {dataset_path}")

    X = data["X"]
    columns_raw = data.get("columns", None)
    if columns_raw is None:
        columns = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        columns = [str(c) for c in list(columns_raw)]

    if not isinstance(X, np.ndarray):
        raise ValueError("X 必须为 numpy.ndarray")

    return X.astype(float), columns


def _load_npy_dataset(dataset_path: str) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")

    payload = np.load(dataset_path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
        payload = payload.item()
    if not isinstance(payload, dict):
        raise ValueError(f"NPY 文件格式不正确: {dataset_path}")

    if "X" not in payload:
        raise ValueError(f"NPY 文件缺少 'X' 键: {dataset_path}")

    X = np.asarray(payload["X"], dtype=float)
    columns_raw = payload.get("columns")
    if columns_raw is None:
        columns = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        columns = [str(col) for col in list(np.asarray(columns_raw).ravel())]

    meta_columns = payload.get("meta_columns")
    meta_data_raw = payload.get("meta_data") or {}
    meta_data: Dict[str, np.ndarray] = {}

    if meta_columns is not None:
        meta_column_list = [str(col) for col in list(np.asarray(meta_columns).ravel())]
    else:
        meta_column_list = list(meta_data_raw.keys())

    for col in meta_column_list:
        values = meta_data_raw.get(col)
        if values is None:
            continue
        arr = np.asarray(values, dtype=object)
        if arr.shape and arr.shape[0] != X.shape[0]:
            raise ValueError(
                f"元信息列 {col} 的长度 ({arr.shape[0]}) 与特征矩阵不一致 ({X.shape[0]})"
            )
        meta_data[col] = arr

    return X.astype(float), columns, meta_data


def _ensure_dirs(results_dir: str, models_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)


def _load_feature_frames(paths: List[str], workers: int) -> List[pd.DataFrame]:
    if not paths:
        return []

    def _read_csv(path: str) -> Tuple[str, pd.DataFrame]:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="gbk")
        return path, df

    frames: List[Tuple[str, pd.DataFrame]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_read_csv, path): path for path in paths}
        for future in as_completed(futures):
            frames.append(future.result())

    frames.sort(key=lambda item: item[0])
    return [df for _, df in frames]


def _sanitize_ratio(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    if ratio <= 0:
        return None
    return float(np.clip(ratio, 0.01, 1.0))


def _build_feature_weights(
    feature_columns: List[str],
    importances: Dict[str, float],
    *,
    ratio: Optional[float],
    min_weight: float = 0.25,
) -> Optional[Dict[str, float]]:
    if not importances:
        return None
    ratio = _sanitize_ratio(ratio)
    if ratio is None:
        return None

    sorted_features = sorted(
        ((feat, float(val)) for feat, val in importances.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not sorted_features:
        return None

    keep_count = max(1, int(round(len(feature_columns) * ratio)))
    selected = {name for name, _ in sorted_features[:keep_count]}
    if not selected:
        return None

    max_importance = sorted_features[0][1] or 1.0
    weights: Dict[str, float] = {}
    for col in feature_columns:
        score = float(importances.get(col, 0.0))
        if col in selected:
            weights[col] = 1.0
        else:
            weights[col] = float(np.clip(score / max_importance, min_weight, 1.0))
    return weights


def _summarize_errors(errors: np.ndarray) -> Dict[str, float]:
    if errors.size == 0:
        return {}
    return {
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.quantile(errors, 0.9)),
        "p95": float(np.quantile(errors, 0.95)),
        "std": float(np.std(errors)),
    }


def _export_active_learning_candidates(
    df: pd.DataFrame,
    *,
    risk_score: np.ndarray,
    vote_ratio: np.ndarray,
    anomaly_score: np.ndarray,
    pseudo_labels: Optional[np.ndarray],
    output_path: str,
    max_candidates: int = 200,
) -> Optional[str]:
    if df.empty or risk_score.size == 0:
        return None
    uncertainty = np.abs(risk_score - 0.5)
    order = np.argsort(uncertainty)
    if order.size == 0:
        return None
    top_n = order[: min(max_candidates, order.size)]
    subset = df.iloc[top_n].copy()
    subset["risk_score"] = risk_score[top_n]
    subset["anomaly_score"] = anomaly_score[top_n]
    subset["vote_ratio"] = vote_ratio[top_n]
    subset["active_learning_uncertainty"] = uncertainty[top_n]
    if pseudo_labels is not None and pseudo_labels.size == len(df):
        subset["semi_supervised_label"] = pseudo_labels[top_n]
    try:
        subset.to_csv(output_path, index=False, encoding="utf-8")
        return output_path
    except Exception as exc:
        logger.warning("Failed to export active learning candidates: %s", exc)
        return None


def _prepare_feature_frame(
    df: pd.DataFrame, feature_columns_hint: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """提取用于训练的特征列，并保持原始顺序。"""

    candidate_cols: List[str] = []
    if feature_columns_hint:
        missing = [col for col in feature_columns_hint if col not in df.columns]
        if missing:
            raise RuntimeError(
                "训练数据缺少以下预期特征列，请重新检查预处理输出: "
                + ", ".join(missing)
            )
        candidate_cols = list(feature_columns_hint)
    else:
        for col in df.columns:
            if col in META_COLUMNS:
                continue
            candidate_cols.append(col)

    if not candidate_cols:
        raise RuntimeError("未发现可用于训练的特征列。")

    feature_df = df.loc[:, candidate_cols].copy()
    return feature_df, candidate_cols


def _train_from_dataframe(
    df: pd.DataFrame,
    *,
    results_dir: str,
    models_dir: str,
    contamination: float,
    base_estimators: int,
    progress_cb=None,
    group_column: Optional[str] = None,
    feature_columns_hint: Optional[List[str]] = None,
    rbf_components: Optional[int] = None,
    rbf_gamma: Optional[float] = None,
    fill_values: Optional[Dict[str, object]] = None,
    categorical_maps: Optional[Dict[str, Dict[str, object]]] = None,
    fusion_alpha: float = 0.5,
    enable_supervised_fusion: bool = True,
    feature_selection_ratio: Optional[float] = None,
) -> dict:
    if df.empty:
        raise RuntimeError("训练数据为空。")

    working_df = df.copy()
    ground_truth_column, ground_truth_raw = _extract_ground_truth(working_df)
    ground_truth: Optional[np.ndarray] = None
    ground_truth_mask: Optional[np.ndarray] = None
    ground_truth_labels: Optional[np.ndarray] = None
    if ground_truth_raw is not None:
        arr = np.asarray(ground_truth_raw, dtype=float)
        labeled_mask = np.isfinite(arr) & (arr >= 0)
        if labeled_mask.any():
            labeled_values = np.where(arr[labeled_mask] > 0, 1.0, 0.0)
            arr = arr.astype(float)
            arr[~labeled_mask] = np.nan
            arr[labeled_mask] = labeled_values
            ground_truth = arr
            ground_truth_mask = labeled_mask.astype(bool)
            ground_truth_labels = labeled_values.astype(int)

    feature_df, feature_columns = _prepare_feature_frame(
        working_df, feature_columns_hint=feature_columns_hint
    )

    adaptive_contamination = float(contamination)
    if ground_truth is not None and ground_truth_labels is not None and ground_truth_labels.size:
        anomaly_ratio = float(np.mean(ground_truth_labels))
        if anomaly_ratio > 0.0:
            adaptive_contamination = float(
                np.clip(anomaly_ratio * 1.2 + 1e-3, 0.001, 0.4)
            )

    adaptive_neighbors = int(max(15, min(120, np.sqrt(len(working_df)) * 2)))
    detector = EnsembleAnomalyDetector(
        contamination=adaptive_contamination,
        n_estimators=max(256, base_estimators * 4),
        n_neighbors=adaptive_neighbors,
        random_state=42,
    )

    fill_values_clean: Dict[str, object] = (
        {str(k): v for k, v in (fill_values or {}).items()}
        if fill_values
        else {}
    )
    categorical_maps_clean: Dict[str, Dict[str, object]] = (
        {str(k): dict(v) if isinstance(v, dict) else {} for k, v in (categorical_maps or {}).items()}
        if categorical_maps
        else {}
    )
    categorical_fill_defaults = {key: -1 for key in categorical_maps_clean.keys()}
    column_fill_values = dict(categorical_fill_defaults)
    column_fill_values.update(fill_values_clean)

    aligner = FeatureAligner(
        feature_columns,
        fill_value=0.0,
        column_fill_values=column_fill_values,
    )
    preprocessor = PreprocessPipeline(
        feature_order=feature_columns,
        fill_value=0.0,
        fill_values=fill_values_clean,
        categorical_maps=categorical_maps_clean,
        aligner_in_pipeline=True,
    )
    feature_df_numeric = feature_df.loc[:, feature_columns].copy()

    ratio_clean = _sanitize_ratio(feature_selection_ratio)
    feature_weights: Optional[Dict[str, float]] = None
    feature_weight_info: Optional[Dict[str, object]] = None
    if ratio_clean is not None:
        numeric_for_var = feature_df_numeric.apply(pd.to_numeric, errors="coerce")
        var_series = numeric_for_var.var(axis=0, skipna=True, ddof=0)
        variance_scores = {
            col: float(var_series.get(col, 0.0)) for col in feature_columns
        }
        candidate_weights = _build_feature_weights(
            feature_columns,
            variance_scores,
            ratio=ratio_clean,
            min_weight=0.15,
        )
        if candidate_weights:
            feature_weights = candidate_weights
            feature_weight_info = {
                "strategy": "variance",
                "ratio": ratio_clean,
                "selected_features": [
                    col for col, w in feature_weights.items() if w >= 0.999
                ],
            }
            logger.info(
                "Applying variance-based feature weighting: keep_ratio=%.2f selected=%d/%d",
                ratio_clean,
                len(feature_weight_info["selected_features"]),
                len(feature_columns),
            )

    data_quality_report: Dict[str, object] = {}
    empty_columns = [
        col
        for col in feature_columns
        if feature_df_numeric[col].dropna().empty
    ]
    constant_columns = [
        col
        for col in feature_columns
        if feature_df_numeric[col].nunique(dropna=True) <= 1
    ]
    if empty_columns:
        logger.warning("Detected empty columns during training: %s", empty_columns[:20])
    if constant_columns:
        logger.warning("Detected constant columns during training: %s", constant_columns[:20])

    winsor_bounds: Dict[str, Dict[str, float]] = {}
    numeric_columns = [
        col for col in feature_columns if is_numeric_dtype(feature_df_numeric[col])
    ]
    if numeric_columns:
        lower_bounds = feature_df_numeric[numeric_columns].quantile(0.001)
        upper_bounds = feature_df_numeric[numeric_columns].quantile(0.999)
        for col in numeric_columns:
            lower = lower_bounds.get(col)
            upper = upper_bounds.get(col)
            if pd.isna(lower) or pd.isna(upper) or lower >= upper:
                continue
            series = feature_df_numeric[col]
            clipped = series.clip(lower, upper)
            if not clipped.equals(series):
                feature_df_numeric[col] = clipped
                winsor_bounds[col] = {
                    "lower": float(lower),
                    "upper": float(upper),
                }
    if winsor_bounds:
        logger.info(
            "Applied winsorization to %d numeric columns", len(winsor_bounds)
        )
    data_quality_report["empty_columns"] = empty_columns
    data_quality_report["constant_columns"] = constant_columns
    data_quality_report["winsorized_columns"] = winsor_bounds

    sample_count = len(feature_df_numeric)
    base_dim = max(len(feature_columns), 1)
    auto_components = int(
        np.clip(np.sqrt(max(1, min(sample_count, base_dim))) * 40, 600, 2000)
    )
    representation_shapes: Dict[str, List[int]] = {
        "base": [int(sample_count), int(base_dim)]
    }
    if rbf_components is not None and rbf_components > 0:
        used_components = int(rbf_components)
    else:
        used_components = auto_components
    auto_gamma = float(np.clip(1.0 / np.sqrt(float(base_dim)), 0.05, 0.5))
    gamma_source = "manual" if rbf_gamma and rbf_gamma > 0 else "auto"
    used_gamma = float(rbf_gamma) if gamma_source == "manual" else auto_gamma
    fusion_alpha = float(np.clip(fusion_alpha if fusion_alpha is not None else 0.5, 0.0, 1.0))
    fusion_requested = bool(enable_supervised_fusion)
    fusion_enabled = fusion_requested
    fusion_auto_enabled = False
    if (
        ground_truth_mask is not None
        and ground_truth_mask.any()
        and not fusion_requested
    ):
        fusion_auto_enabled = True
        fusion_enabled = True
        logger.info(
            "Detected ground truth column '%s', automatically enabling semi-supervised fusion.",
            ground_truth_column or "unknown",
        )
    elif ground_truth_mask is not None and ground_truth_mask.any():
        fusion_enabled = True
    fusion_source: Optional[str] = None

    feature_weighter = FeatureWeighter(feature_weights or {})
    deep_latent_dim = int(np.clip(len(feature_columns) // 2, 8, 128))
    deep_extractor = DeepFeatureExtractor(
        latent_dim=deep_latent_dim,
        random_state=42,
        max_epochs=25,
        batch_size=256,
    )

    pipeline_steps = [
        ("aligner", aligner),
        ("preprocessor", preprocessor),
        ("feature_weighter", feature_weighter),
        ("variance_filter", VarianceThreshold(threshold=1e-6)),
        ("scaler", StandardScaler()),
        ("deep_features", deep_extractor),
        (
            "gaussianizer",
            QuantileTransformer(
                output_distribution="normal",
                subsample=200000,
                random_state=42,
            ),
        ),
        (
            "feature_expander",
            RBFSampler(
                n_components=used_components,
                gamma=used_gamma,
                random_state=42,
            ),
        ),
        ("detector", detector),
    ]

    pipeline = Pipeline(pipeline_steps)

    logger.info(
        "Starting unsupervised training: rows=%d features=%d contamination=%.4f (adaptive=%.4f) rbf_components=%d rbf_gamma=%.3f [%s]",
        len(feature_df_numeric),
        len(feature_columns),
        contamination,
        adaptive_contamination,
        used_components,
        used_gamma,
        gamma_source,
    )

    if progress_cb:
        progress_cb(70)

    supervised_proba = None
    if ground_truth is not None:
        pipeline.fit(feature_df_numeric, ground_truth)
    else:
        pipeline.fit(feature_df_numeric)
    try:
        pipeline.feature_names_in_ = np.asarray(feature_columns)
    except Exception:
        pass
    detector = pipeline.named_steps["detector"]
    preprocessor = pipeline.named_steps.get("preprocessor", preprocessor)
    pre_detector = Pipeline(pipeline.steps[:-1])
    transformed_features: Optional[np.ndarray] = None
    deep_errors: Optional[np.ndarray] = None
    deep_feature_info: Dict[str, object] = {}

    deep_step = pipeline.named_steps.get("deep_features")
    if deep_step is not None:
        deep_feature_info["latent_dim"] = int(
            getattr(deep_step, "latent_dim_", getattr(deep_step, "latent_dim", 0))
            or deep_latent_dim
        )
        deep_feature_info["training_loss"] = float(
            getattr(deep_step, "training_loss_", 0.0) or 0.0
        )
        try:
            gaussianizer_idx = next(
                i for i, (name, _) in enumerate(pipeline.steps) if name == "gaussianizer"
            )
            pre_gaussianizer = Pipeline(pipeline.steps[:gaussianizer_idx])
            deep_output = pre_gaussianizer.transform(feature_df_numeric)
            if isinstance(deep_output, np.ndarray) and deep_output.ndim == 2:
                representation_shapes["deep_augmented"] = [
                    int(deep_output.shape[0]),
                    int(deep_output.shape[1]),
                ]
                error_idx = getattr(deep_step, "error_index_", None)
                if error_idx is None:
                    error_idx = deep_output.shape[1] - 1
                deep_errors = np.asarray(deep_output[:, int(error_idx)], dtype=float)
                latent_slice = getattr(deep_step, "latent_slice_", None)
                if isinstance(latent_slice, slice):
                    representation_shapes["latent"] = [
                        int(deep_output.shape[0]),
                        int(latent_slice.stop - latent_slice.start),
                    ]
                deep_summary = _summarize_errors(deep_errors)
                if deep_summary:
                    deep_feature_info["reconstruction_error"] = deep_summary
        except Exception as exc:
            logger.warning("Failed to derive deep feature representation: %s", exc)

    supervised_metrics = None
    svd_info: Optional[Dict[str, object]] = None
    feature_importances_topk: Optional[List[Dict[str, object]]] = None
    permutation_topk: Optional[List[Dict[str, object]]] = None
    if ground_truth is not None:
        transformed = pre_detector.transform(feature_df_numeric)
        transformed_features = transformed
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)
        if transformed.ndim == 2:
            representation_shapes["expanded"] = [
                int(transformed.shape[0]),
                int(transformed.shape[1]),
            ]

        svd = None
        reduced = transformed
        if transformed.shape[1] > 512:
            target_dim = min(512, max(64, transformed.shape[1] // 4))
            svd = TruncatedSVD(n_components=target_dim, random_state=42)
            reduced = svd.fit_transform(transformed)
            representation_shapes["reduced"] = [
                int(reduced.shape[0]),
                int(reduced.shape[1]),
            ]
            if hasattr(svd, "explained_variance_ratio_"):
                ratios = np.asarray(svd.explained_variance_ratio_, dtype=float)
                top_k = min(20, ratios.size)
                svd_info = {
                    "components": int(ratios.size),
                    "top_variance_ratio": [float(v) for v in ratios[:top_k]],
                    "cumulative_top": float(ratios[:top_k].sum()) if top_k else 0.0,
                    "total_variance": float(ratios.sum()),
                }

        can_train_supervised = (
            ground_truth_mask is not None
            and ground_truth_labels is not None
            and np.unique(ground_truth_labels).size >= 2
        )
        if can_train_supervised:
            mask_series = pd.Series(ground_truth_mask, index=feature_df_numeric.index)
            labeled_reduced = reduced[ground_truth_mask]
            labeled_truth = ground_truth_labels

            supervised_model = HistGradientBoostingClassifier(
                max_depth=5,
                learning_rate=0.08,
                max_iter=200,
                l2_regularization=1.0,
                early_stopping=True,
                random_state=42,
            )
            supervised_model.fit(labeled_reduced, labeled_truth)
            supervised_proba_all = supervised_model.predict_proba(reduced)[:, 1]
            supervised_proba = supervised_proba_all
            thr_candidates = np.linspace(0.1, 0.9, 41)
            best_thr = 0.5
            best_f1 = -1.0
            best_f05 = -1.0
            best_prec = 0.0
            best_rec = 0.0
            labeled_scores = supervised_proba_all[ground_truth_mask]
            for thr in thr_candidates:
                preds = (labeled_scores >= thr).astype(int)
                precision = precision_score(labeled_truth, preds, zero_division=0)
                recall = recall_score(labeled_truth, preds, zero_division=0)
                f1 = f1_score(labeled_truth, preds, zero_division=0)
                f05 = fbeta_score(labeled_truth, preds, beta=0.5, zero_division=0)
                if (f05 > best_f05 + 1e-12) or (
                    abs(f05 - best_f05) <= 1e-12 and f1 > best_f1
                ):
                    best_f1 = f1
                    best_f05 = f05
                    best_thr = thr
                    best_prec = precision
                    best_rec = recall
            detector.supervised_model_ = supervised_model
            detector.supervised_threshold_ = float(best_thr)
            detector.supervised_input_dim_ = reduced.shape[1]
            detector.supervised_projector_ = svd
            detector.last_supervised_scores_ = supervised_proba_all.astype(float)
            supervised_metrics = {
                "precision": float(best_prec),
                "recall": float(best_rec),
                "f1": float(best_f1),
                "f0.5": float(best_f05),
                "threshold": float(best_thr),
            }

            try:
                raw_supervised = HistGradientBoostingClassifier(
                    max_depth=5,
                    learning_rate=0.08,
                    max_iter=200,
                    l2_regularization=1.0,
                    early_stopping=True,
                    random_state=137,
                )
                labeled_features = feature_df_numeric.loc[mask_series, feature_columns]
                labeled_truth_series = pd.Series(labeled_truth, index=labeled_features.index)
                raw_supervised.fit(labeled_features, labeled_truth_series.to_numpy())
                importances = np.asarray(raw_supervised.feature_importances_, dtype=float)
                if importances.size:
                    order = np.argsort(importances)[::-1]
                    top_limit = min(20, importances.size)
                    feature_importances_topk = [
                        {
                            "feature": feature_columns[idx],
                            "importance": float(importances[idx]),
                        }
                        for idx in order[:top_limit]
                        if importances[idx] > 0
                    ]

                try:
                    perm_features = labeled_features
                    max_perm_samples = 5000
                    if len(perm_features) > max_perm_samples:
                        rng = np.random.default_rng(42)
                        sample_idx = rng.choice(
                            perm_features.index.to_numpy(), size=max_perm_samples, replace=False
                        )
                        perm_features = perm_features.loc[sample_idx]
                        perm_labels = labeled_truth_series.loc[sample_idx].to_numpy()
                    else:
                        perm_labels = labeled_truth_series.to_numpy()
                    perm_result = permutation_importance(
                        raw_supervised,
                        perm_features,
                        perm_labels,
                        n_repeats=5,
                        random_state=42,
                        n_jobs=-1,
                    )
                    perm_scores = perm_result.importances_mean
                    order = np.argsort(perm_scores)[::-1]
                    permutation_topk = [
                        {
                            "feature": feature_columns[idx],
                            "importance": float(perm_scores[idx]),
                        }
                        for idx in order[: min(20, len(order))]
                        if perm_scores[idx] > 0
                    ]
                except Exception as exc:
                    logger.warning("Failed to compute permutation importance: %s", exc)

            except Exception as exc:
                logger.warning("Failed to compute feature importances: %s", exc)
        else:
            logger.info("Skipping supervised fine-tuning due to insufficient labeled samples.")

    scores = None
    if detector.last_combined_scores_ is not None:
        scores = detector.last_combined_scores_
    elif detector.fit_decision_scores_ is not None:
        scores = detector.fit_decision_scores_
    else:
        if transformed_features is None:
            transformed_features = pre_detector.transform(feature_df_numeric)
        if (
            transformed_features is not None
            and transformed_features.ndim == 2
            and "expanded" not in representation_shapes
        ):
            representation_shapes["expanded"] = [
                int(transformed_features.shape[0]),
                int(transformed_features.shape[1]),
            ]
        scores = detector.score_samples(transformed_features)
    scores = np.asarray(scores, dtype=float)

    feature_expander = pipeline.named_steps.get("feature_expander")
    expanded_dim = None
    if feature_expander is not None and hasattr(feature_expander, "n_components"):
        expanded_dim = int(getattr(feature_expander, "n_components"))
    elif transformed_features is not None:
        expanded_dim = int(transformed_features.shape[1])
    if expanded_dim is not None and "expanded" not in representation_shapes:
        representation_shapes["expanded"] = [int(sample_count), int(expanded_dim)]

    preds = pipeline.predict(feature_df_numeric)
    is_malicious = (preds == -1).astype(int)

    vote_ratio = None
    if detector.last_vote_ratio_ is not None:
        vote_ratio = detector.last_vote_ratio_
    elif detector.fit_votes_:
        vote_ratio = np.vstack([np.where(v == -1, 1.0, 0.0) for v in detector.fit_votes_.values()]).mean(axis=0)
    else:
        vote_ratio = np.ones_like(is_malicious, dtype=float)
    vote_ratio = np.asarray(vote_ratio, dtype=float)

    effective_contamination = float(detector.contamination)
    threshold = detector.threshold_ if detector.threshold_ is not None else float(np.quantile(scores, effective_contamination))
    vote_threshold = detector.vote_threshold_ if detector.vote_threshold_ is not None else float(np.mean(vote_ratio))
    vote_threshold = float(np.clip(vote_threshold, 0.0, 1.0))
    training_ratio_value = (
        float(detector.training_anomaly_ratio_)
        if detector.training_anomaly_ratio_ is not None
        else float(is_malicious.mean())
    )

    score_std = float(np.std(scores) or 1.0)
    conf_from_score = 1.0 / (1.0 + np.exp((scores - threshold) / (score_std + 1e-6)))
    vote_component = np.clip((vote_ratio - vote_threshold) / max(1e-6, (1.0 - vote_threshold)), 0.0, 1.0)
    risk_score = np.clip(0.6 * conf_from_score + 0.4 * vote_component, 0.0, 1.0)

    if deep_errors is not None and deep_errors.size == risk_score.size:
        ae_min = float(np.min(deep_errors))
        ae_range = float(np.max(deep_errors) - ae_min)
        ae_norm = (deep_errors - ae_min) / max(ae_range, 1e-6)
        risk_score = np.clip(0.5 * risk_score + 0.5 * ae_norm, 0.0, 1.0)
        if deep_feature_info is not None:
            deep_feature_info.setdefault("reconstruction_error", _summarize_errors(deep_errors))

    if fusion_enabled:
        fusion_candidates = [
            ("supervised_model", getattr(detector, "last_supervised_scores_", None)),
            ("calibration", getattr(detector, "last_calibrated_scores_", None)),
            ("supervised_proba", supervised_proba),
        ]
        for label, candidate in fusion_candidates:
            if candidate is None:
                continue
            extra = np.asarray(candidate, dtype=float)
            if extra.shape != risk_score.shape:
                continue
            extra = np.clip(extra, 0.0, 1.0)
            risk_score = np.clip(
                fusion_alpha * risk_score + (1.0 - fusion_alpha) * extra,
                0.0,
                1.0,
            )
            fusion_source = label
            break

    anomaly_confidence = risk_score.copy()

    working_df["anomaly_score"] = scores
    working_df["vote_ratio"] = vote_ratio
    working_df["anomaly_confidence"] = anomaly_confidence
    working_df["risk_score"] = risk_score
    working_df["is_malicious"] = is_malicious
    if deep_errors is not None and deep_errors.size == len(working_df):
        working_df["ae_reconstruction_error"] = deep_errors

    pseudo_labels = getattr(detector, "pseudo_labels_", None)
    pseudo_origins = getattr(detector, "pseudo_label_origins_", None)
    if pseudo_labels is not None and len(pseudo_labels) == len(working_df):
        working_df["semi_supervised_label"] = pseudo_labels
        if pseudo_origins is not None and len(pseudo_origins) == len(working_df):
            working_df["semi_label_origin"] = pseudo_origins
    pseudo_summary = getattr(detector, "pseudo_label_summary_", None)

    active_learning_csv = _export_active_learning_candidates(
        working_df,
        risk_score=risk_score,
        vote_ratio=vote_ratio,
        anomaly_score=scores,
        pseudo_labels=pseudo_labels if pseudo_labels is not None else None,
        output_path=os.path.join(results_dir, "active_learning_candidates.csv"),
    )

    results_csv = os.path.join(results_dir, "iforest_results.csv")
    working_df.to_csv(results_csv, index=False, encoding="utf-8")

    if group_column and group_column in working_df.columns:
        summary = (
            working_df.groupby(group_column)
            .agg(
                total_packets=(group_column, "count"),
                malicious_packets=("is_malicious", "sum"),
                malicious_ratio=("is_malicious", "mean"),
            )
            .reset_index()
            .sort_values("malicious_ratio", ascending=False)
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "source": group_column or "all",
                    "total_packets": len(working_df),
                    "malicious_packets": int(is_malicious.sum()),
                    "malicious_ratio": float(is_malicious.mean()),
                }
            ]
        )

    summary_csv = os.path.join(results_dir, "summary_by_file.csv")
    summary.to_csv(summary_csv, index=False, encoding="utf-8")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_name = f"iforest_pipeline_{timestamp}.joblib"
    pipeline_path = os.path.join(models_dir, pipeline_name)
    dump(pipeline, pipeline_path)

    preprocessor_path = os.path.join(models_dir, f"preprocessor_{timestamp}.joblib")
    dump(preprocessor, preprocessor_path)

    latest_pipeline_path = os.path.join(models_dir, "latest_iforest_pipeline.joblib")
    try:
        shutil.copy2(pipeline_path, latest_pipeline_path)
    except Exception:
        dump(pipeline, latest_pipeline_path)

    latest_preprocessor_path = os.path.join(models_dir, "latest_preprocessor.joblib")
    try:
        shutil.copy2(preprocessor_path, latest_preprocessor_path)
    except Exception:
        dump(preprocessor, latest_preprocessor_path)

    scaler_path = os.path.join(models_dir, "scaler.joblib")
    dump(pipeline.named_steps["scaler"], scaler_path)
    gaussianizer_path = os.path.join(models_dir, "gaussianizer.joblib")
    dump(pipeline.named_steps["gaussianizer"], gaussianizer_path)
    model_path = os.path.join(models_dir, "isoforest.joblib")
    base_iforest = detector.detectors_.get("iforest")
    if base_iforest:
        dump(base_iforest.estimator, model_path)
    else:
        dump(detector, model_path)

    preprocessor_metadata = {}
    try:
        preprocessor_metadata = preprocessor.to_metadata()
    except Exception:
        preprocessor_metadata = {}

    feature_order = list(getattr(preprocessor, "feature_order", feature_columns))
    fill_values = dict(getattr(preprocessor, "fill_values", {}) or {})
    categorical_maps = dict(getattr(preprocessor, "categorical_maps", {}) or {})
    quantile_points = (0.01, 0.05, 0.5, 0.9)
    score_quantiles: Dict[str, float] = {}
    if scores.size:
        for q in quantile_points:
            try:
                score_quantiles[str(q)] = float(np.quantile(scores, q))
            except Exception:
                continue

    score_histogram = None
    if scores.size:
        try:
            hist_counts, hist_bins = np.histogram(scores, bins=min(60, max(10, int(np.sqrt(scores.size)))))
            score_histogram = {
                "bins": [float(v) for v in hist_bins.tolist()],
                "counts": [int(v) for v in hist_counts.tolist()],
            }
        except Exception:
            score_histogram = None

    feature_hash = hashlib.sha256("\n".join(feature_columns).encode("utf-8")).hexdigest()
    feature_list_payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "timestamp": timestamp,
        "feature_columns": feature_columns,
        "feature_hash": feature_hash,
    }
    feature_list_name = f"iforest_features_{timestamp}.json"
    feature_list_path = os.path.join(models_dir, feature_list_name)
    with open(feature_list_path, "w", encoding="utf-8") as fh:
        json.dump(feature_list_payload, fh, ensure_ascii=False, indent=2)

    latest_feature_list_path = os.path.join(models_dir, "latest_feature_list.json")
    with open(latest_feature_list_path, "w", encoding="utf-8") as fh:
        json.dump(feature_list_payload, fh, ensure_ascii=False, indent=2)

    metadata = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "timestamp": timestamp,
        "contamination": effective_contamination,
        "requested_contamination": float(contamination),
        "base_estimators": base_estimators,
        "feature_columns": feature_columns,
        "feature_order": feature_order,
        "expanded_dim": int(expanded_dim) if expanded_dim is not None else None,
        "threshold": float(threshold),
        "score_std": float(score_std),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_quantiles": score_quantiles,
        "score_histogram": score_histogram,
        "vote_mean": float(np.mean(vote_ratio)),
        "vote_threshold": float(vote_threshold),
        "threshold_breakdown": detector.threshold_breakdown_,
        "detectors": list(detector.detectors_.keys()),
        "training_anomaly_ratio": training_ratio_value,
        "estimated_precision": float(anomaly_confidence[is_malicious == 1].mean() if is_malicious.any() else 0.0),
        "estimated_anomaly_ratio": float(is_malicious.mean()),
        "results_csv": results_csv,
        "summary_csv": summary_csv,
        "active_learning_csv": active_learning_csv,
        "gaussianizer_path": gaussianizer_path,
        "projection_dim": int(detector.projected_dim_) if detector.projected_dim_ is not None else None,
        "projected_dim": int(detector.projected_dim_) if detector.projected_dim_ is not None else None,
        "preprocessor": preprocessor_metadata,
        "fill_value": float(getattr(preprocessor, "fill_value", 0.0)),
        "fill_values": fill_values,
        "categorical_maps": categorical_maps,
        "feature_names_in": feature_columns,
        "rbf_components": used_components,
        "rbf_n_components": used_components,
        "rbf_gamma": used_gamma,
        "rbf_gamma_auto": auto_gamma,
        "rbf_gamma_source": gamma_source,
        "fusion_enabled": fusion_enabled,
        "fusion_alpha": fusion_alpha,
        "fusion_source": fusion_source,
        "fusion_auto_enabled": fusion_auto_enabled,
        "representation_shapes": representation_shapes,
        "svd_info": svd_info,
        "data_quality": data_quality_report,
        "preprocessor_path": preprocessor_path,
        "preprocessor_latest": latest_preprocessor_path,
        "pipeline_path": pipeline_path,
        "pipeline_latest": latest_pipeline_path,
        "model_path": model_path,
        "feature_hash": feature_hash,
        "feature_list_path": feature_list_path,
        "feature_list_latest": latest_feature_list_path,
    }

    if feature_weight_info and feature_weights:
        weight_payload = dict(feature_weight_info)
        weight_payload["weights"] = feature_weights
        metadata["feature_weighting"] = weight_payload

    if deep_feature_info:
        metadata["deep_features"] = deep_feature_info

    if pseudo_summary:
        metadata["pseudo_labels"] = pseudo_summary

    if detector.calibration_report_ is not None:
        metadata["calibration"] = detector.calibration_report_
        metadata["calibration_threshold"] = float(
            detector.calibration_threshold_ if detector.calibration_threshold_ is not None else 0.5
        )

    if supervised_metrics is not None:
        metadata["supervised"] = supervised_metrics
    if feature_importances_topk:
        metadata["feature_importances_topk"] = feature_importances_topk
    if permutation_topk:
        metadata["permutation_importance_topk"] = permutation_topk

    if (
        ground_truth is not None
        and ground_truth_mask is not None
        and ground_truth_labels is not None
        and ground_truth_labels.size
    ):
        labeled_preds = is_malicious[ground_truth_mask]
        labeled_truth = ground_truth_labels
        precision = precision_score(labeled_truth, labeled_preds, zero_division=0)
        recall = recall_score(labeled_truth, labeled_preds, zero_division=0)
        f1 = f1_score(labeled_truth, labeled_preds, zero_division=0)
        acc = accuracy_score(labeled_truth, labeled_preds)
        cm = confusion_matrix(labeled_truth, labeled_preds).tolist()
        metadata["ground_truth_column"] = ground_truth_column
        metadata["ground_truth_ratio"] = float(np.mean(labeled_truth))
        metadata["ground_truth_samples"] = int(labeled_truth.size)
        metadata["evaluation"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(acc),
            "confusion_matrix": cm,
        }

    metadata_path = os.path.join(models_dir, f"iforest_metadata_{timestamp}.json")
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    latest_meta_path = os.path.join(models_dir, "latest_iforest_metadata.json")
    with open(latest_meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    logger.info(
        "Training finished: threshold=%.6f vote_threshold=%.3f anomalies=%d/%d",
        float(threshold),
        float(vote_threshold),
        int(is_malicious.sum()),
        len(is_malicious),
    )

    if progress_cb:
        progress_cb(100)

    result_payload = {
        "results_csv": results_csv,
        "summary_csv": summary_csv,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "pipeline_path": pipeline_path,
        "pipeline_latest": latest_pipeline_path,
        "metadata_path": metadata_path,
        "metadata_latest": latest_meta_path,
        "preprocessor_path": preprocessor_path,
        "preprocessor_latest": latest_preprocessor_path,
        "packets": len(working_df),
        "flows": len(working_df),
        "malicious": int(is_malicious.sum()),
        "contamination": effective_contamination,
        "requested_contamination": float(contamination),
        "threshold": float(threshold),
        "vote_threshold": float(vote_threshold),
        "estimated_precision": metadata["estimated_precision"],
        "threshold_breakdown": detector.threshold_breakdown_,
        "feature_columns": feature_columns,
        "expanded_dim": expanded_dim,
        "training_anomaly_ratio": training_ratio_value,
        "gaussianizer_path": gaussianizer_path,
        "ground_truth_column": ground_truth_column,
        "evaluation": metadata.get("evaluation"),
        "supervised": supervised_metrics,
        "projection_dim": metadata.get("projection_dim"),
        "preprocessor": preprocessor_metadata,
        "rbf_components": used_components,
        "rbf_n_components": used_components,
        "rbf_gamma": used_gamma,
        "rbf_gamma_auto": auto_gamma,
        "rbf_gamma_source": gamma_source,
        "fusion_enabled": fusion_enabled,
        "fusion_alpha": fusion_alpha,
        "fusion_source": fusion_source,
        "fusion_auto_enabled": fusion_auto_enabled,
        "representation_shapes": representation_shapes,
        "svd_info": svd_info,
        "data_quality": data_quality_report,
        "active_learning_csv": active_learning_csv,
        "score_histogram": score_histogram,
    }

    if feature_weight_info and feature_weights:
        weight_payload = dict(feature_weight_info)
        weight_payload["weights"] = feature_weights
        result_payload["feature_weighting"] = weight_payload

    if deep_feature_info:
        result_payload["deep_features"] = deep_feature_info

    if pseudo_summary:
        result_payload["pseudo_labels"] = pseudo_summary

    if feature_importances_topk:
        result_payload["feature_importances_topk"] = feature_importances_topk
    if permutation_topk:
        result_payload["permutation_importance_topk"] = permutation_topk
    result_payload["feature_list_path"] = feature_list_path
    result_payload["feature_list_latest"] = latest_feature_list_path
    result_payload["feature_hash"] = feature_hash

    return result_payload


def _train_from_preprocessed_csv(
    dataset_path: str,
    *,
    results_dir: str,
    models_dir: str,
    contamination: float,
    base_estimators: int,
    progress_cb=None,
    rbf_components: Optional[int] = None,
    rbf_gamma: Optional[float] = None,
    fusion_alpha: float = 0.5,
    enable_supervised_fusion: bool = True,
    feature_selection_ratio: Optional[float] = None,
) -> dict:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"预处理数据集不存在: {dataset_path}")

    df = pd.read_csv(dataset_path, encoding="utf-8")
    if df.empty:
        raise RuntimeError("预处理数据集为空，无法训练。")

    manifest_path = _manifest_path_for(dataset_path)
    manifest_col = None
    feature_columns_hint: Optional[List[str]] = None
    fill_values_hint: Optional[Dict[str, object]] = None
    categorical_maps_hint: Optional[Dict[str, Dict[str, object]]] = None

    meta_path = _meta_path_for(dataset_path)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta_payload = json.load(fh)
            if isinstance(meta_payload, dict):
                cols = meta_payload.get("feature_columns")
                if isinstance(cols, list):
                    feature_columns_hint = [str(col) for col in cols]
                fill_values_hint = meta_payload.get("fill_values") or meta_payload.get("fill_strategies")
                if fill_values_hint is not None:
                    fill_values_hint = {
                        str(k): v for k, v in dict(fill_values_hint).items()
                    }
                categorical_maps_raw = meta_payload.get("categorical_maps")
                if isinstance(categorical_maps_raw, dict):
                    categorical_maps_hint = {
                        str(k): dict(v) if isinstance(v, dict) else {}
                        for k, v in categorical_maps_raw.items()
                    }
        except Exception as exc:
            print(f"[WARN] 读取元数据失败 {meta_path}: {exc}")

    if os.path.exists(manifest_path):
        try:
            manifest_df = pd.read_csv(manifest_path, encoding="utf-8")
            expanded = []
            for _, row in manifest_df.iterrows():
                rows = int(row.get("rows", 0))
                source_file = row.get("source_file") or row.get("source_path") or "unknown"
                expanded.extend([source_file] * rows)
            if len(expanded) == len(df):
                df["__source_file__"] = expanded
                manifest_col = "__source_file__"
        except Exception as exc:
            print(f"[WARN] 读取 manifest 失败 {manifest_path}: {exc}")

    if "__source_file__" in df.columns:
        manifest_col = "__source_file__"
    elif "pcap_file" in df.columns:
        manifest_col = "pcap_file"

    if progress_cb:
        progress_cb(40)

    return _train_from_dataframe(
        df,
        results_dir=results_dir,
        models_dir=models_dir,
        contamination=contamination,
        base_estimators=base_estimators,
        progress_cb=progress_cb,
        group_column=manifest_col,
        feature_columns_hint=feature_columns_hint,
        rbf_components=rbf_components,
        rbf_gamma=rbf_gamma,
        fill_values=fill_values_hint,
        categorical_maps=categorical_maps_hint,
        fusion_alpha=fusion_alpha,
        enable_supervised_fusion=enable_supervised_fusion,
        feature_selection_ratio=feature_selection_ratio,
    )


def _train_from_npz(
    dataset_path: str,
    *,
    results_dir: str,
    models_dir: str,
    contamination: float,
    base_estimators: int,
    progress_cb=None,
    rbf_components: Optional[int] = None,
    rbf_gamma: Optional[float] = None,
    fusion_alpha: float = 0.5,
    enable_supervised_fusion: bool = True,
    feature_selection_ratio: Optional[float] = None,
) -> dict:
    X, columns = _load_npz_dataset(dataset_path)
    df = pd.DataFrame(X, columns=columns)

    manifest_path = dataset_path.replace(".npz", "_manifest.csv")
    manifest_col = None
    if os.path.exists(manifest_path):
        try:
            manifest_df = pd.read_csv(manifest_path, encoding="utf-8")
            expanded = []
            for _, row in manifest_df.iterrows():
                rows = int(row.get("rows", 0))
                source_file = row.get("source_file") or row.get("source_path") or "unknown"
                expanded.extend([source_file] * rows)
            if len(expanded) == len(df):
                df["__source_file__"] = expanded
                manifest_col = "__source_file__"
        except Exception as exc:
            print(f"[WARN] 读取 manifest 失败 {manifest_path}: {exc}")

    if progress_cb:
        progress_cb(40)

    fill_values_hint: Optional[Dict[str, object]] = None
    categorical_maps_hint: Optional[Dict[str, Dict[str, object]]] = None
    meta_path = _meta_path_for(dataset_path)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta_payload = json.load(fh)
            if isinstance(meta_payload, dict):
                fill_values_hint = meta_payload.get("fill_values") or meta_payload.get("fill_strategies")
                if fill_values_hint is not None:
                    fill_values_hint = {
                        str(k): v for k, v in dict(fill_values_hint).items()
                    }
                categorical_maps_raw = meta_payload.get("categorical_maps")
                if isinstance(categorical_maps_raw, dict):
                    categorical_maps_hint = {
                        str(k): dict(v) if isinstance(v, dict) else {}
                        for k, v in categorical_maps_raw.items()
                    }
        except Exception as exc:
            print(f"[WARN] 读取元数据失败 {meta_path}: {exc}")

    return _train_from_dataframe(
        df,
        results_dir=results_dir,
        models_dir=models_dir,
        contamination=contamination,
        base_estimators=base_estimators,
        progress_cb=progress_cb,
        group_column=manifest_col,
        feature_columns_hint=list(columns),
        rbf_components=rbf_components,
        rbf_gamma=rbf_gamma,
        fill_values=fill_values_hint,
        categorical_maps=categorical_maps_hint,
        fusion_alpha=fusion_alpha,
        enable_supervised_fusion=enable_supervised_fusion,
        feature_selection_ratio=feature_selection_ratio,
    )


def _train_from_npy(
    dataset_path: str,
    *,
    results_dir: str,
    models_dir: str,
    contamination: float,
    base_estimators: int,
    progress_cb=None,
    rbf_components: Optional[int] = None,
    rbf_gamma: Optional[float] = None,
    fusion_alpha: float = 0.5,
    enable_supervised_fusion: bool = True,
    feature_selection_ratio: Optional[float] = None,
) -> dict:
    X, columns, meta_data = _load_npy_dataset(dataset_path)
    df = pd.DataFrame(X, columns=columns)

    for col, values in meta_data.items():
        try:
            values_len = len(values)
        except TypeError:
            continue
        if values_len == len(df):
            df[col] = pd.Series(values)

    manifest_path = _manifest_path_for(dataset_path)
    manifest_col = None
    if os.path.exists(manifest_path):
        try:
            manifest_df = pd.read_csv(manifest_path, encoding="utf-8")
            expanded = []
            for _, row in manifest_df.iterrows():
                rows = int(row.get("rows", 0))
                source_file = row.get("source_file") or row.get("source_path") or "unknown"
                expanded.extend([source_file] * rows)
            if len(expanded) == len(df):
                df["__source_file__"] = expanded
                manifest_col = "__source_file__"
        except Exception as exc:
            print(f"[WARN] 读取 manifest 失败 {manifest_path}: {exc}")

    if "__source_file__" in df.columns:
        manifest_col = "__source_file__"
    elif "pcap_file" in df.columns:
        manifest_col = "pcap_file"

    feature_columns_hint: Optional[List[str]] = list(columns)
    fill_values_hint: Optional[Dict[str, object]] = None
    categorical_maps_hint: Optional[Dict[str, Dict[str, object]]] = None
    meta_path = _meta_path_for(dataset_path)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta_payload = json.load(fh)
            cols = meta_payload.get("feature_columns")
            if isinstance(cols, list) and cols:
                feature_columns_hint = [str(col) for col in cols]
            fill_values_hint = meta_payload.get("fill_values") or meta_payload.get("fill_strategies")
            if fill_values_hint is not None:
                fill_values_hint = {
                    str(k): v for k, v in dict(fill_values_hint).items()
                }
            categorical_maps_raw = meta_payload.get("categorical_maps")
            if isinstance(categorical_maps_raw, dict):
                categorical_maps_hint = {
                    str(k): dict(v) if isinstance(v, dict) else {}
                    for k, v in categorical_maps_raw.items()
                }
        except Exception as exc:
            print(f"[WARN] 读取元数据失败 {meta_path}: {exc}")

    if progress_cb:
        progress_cb(40)

    return _train_from_dataframe(
        df,
        results_dir=results_dir,
        models_dir=models_dir,
        contamination=contamination,
        base_estimators=base_estimators,
        progress_cb=progress_cb,
        group_column=manifest_col,
        feature_columns_hint=feature_columns_hint,
        rbf_components=rbf_components,
        rbf_gamma=rbf_gamma,
        fill_values=fill_values_hint,
        categorical_maps=categorical_maps_hint,
        fusion_alpha=fusion_alpha,
        enable_supervised_fusion=enable_supervised_fusion,
        feature_selection_ratio=feature_selection_ratio,
    )


def train_unsupervised_on_split(
    split_dir: str,
    results_dir: str,
    models_dir: str,
    contamination: float = 0.05,
    base_estimators: int = 50,
    progress_cb=None,
    workers: int = 4,
    rbf_components: Optional[int] = None,
    rbf_gamma: Optional[float] = None,
    fusion_alpha: float = 0.5,
    enable_supervised_fusion: bool = True,
    feature_selection_ratio: Optional[float] = None,
):
    """
    无监督训练：
    - 支持直接加载预处理 CSV/NPY/NPZ 数据集
    - 对 PCAP 目录进行特征提取（多线程）后再训练
    """

    _ensure_dirs(results_dir, models_dir)

    dataset_path: Optional[str] = None
    pcap_dir: Optional[str] = None
    pcap_files: List[str] = []

    if not split_dir:
        raise FileNotFoundError("未提供训练数据路径")

    if os.path.isfile(split_dir):
        user_selected_path = split_dir
        resolved_dataset = _resolve_dataset_from_auxiliary(user_selected_path)
        if resolved_dataset != user_selected_path:
            if os.path.exists(resolved_dataset):
                split_dir = resolved_dataset
            else:
                raise FileNotFoundError(
                    "未找到与所选 manifest/meta 文件对应的数据集主文件，"
                    "请重新选择预处理输出的 .npy/.npz/.csv 数据文件。"
                )

        if _is_preprocessed_csv(split_dir) or _is_npy(split_dir) or (
            _is_npz(split_dir) and os.path.basename(split_dir).startswith("dataset_preprocessed_")
        ):
            dataset_path = split_dir
        elif _is_npz(split_dir):
            dataset_path = split_dir
        elif split_dir.lower().endswith((".pcap", ".pcapng")):
            pcap_dir = os.path.dirname(split_dir) or os.path.abspath(os.path.join(split_dir, os.pardir))
            pcap_files = [split_dir]
        else:
            raise FileNotFoundError(f"不支持的训练文件: {split_dir}")
    elif os.path.isdir(split_dir):
        dataset_path = _latest_preprocessed_dataset(split_dir)
        if dataset_path is None:
            dataset_path = _latest_npz_in_dir(split_dir)
        if dataset_path is None:
            pcap_dir = split_dir
    else:
        raise FileNotFoundError(f"路径不存在: {split_dir}")

    if dataset_path:
        ext = os.path.splitext(dataset_path)[1].lower()
        if ext == ".csv":
            return _train_from_preprocessed_csv(
                dataset_path,
                results_dir=results_dir,
                models_dir=models_dir,
                contamination=contamination,
                base_estimators=base_estimators,
                progress_cb=progress_cb,
                rbf_components=rbf_components,
                rbf_gamma=rbf_gamma,
                fusion_alpha=fusion_alpha,
                enable_supervised_fusion=enable_supervised_fusion,
                feature_selection_ratio=feature_selection_ratio,
            )
        if ext == ".npy":
            return _train_from_npy(
                dataset_path,
                results_dir=results_dir,
                models_dir=models_dir,
                contamination=contamination,
                base_estimators=base_estimators,
                progress_cb=progress_cb,
                rbf_components=rbf_components,
                rbf_gamma=rbf_gamma,
                fusion_alpha=fusion_alpha,
                enable_supervised_fusion=enable_supervised_fusion,
                feature_selection_ratio=feature_selection_ratio,
            )
        if ext == ".npz":
            return _train_from_npz(
                dataset_path,
                results_dir=results_dir,
                models_dir=models_dir,
                contamination=contamination,
                base_estimators=base_estimators,
                progress_cb=progress_cb,
                rbf_components=rbf_components,
                rbf_gamma=rbf_gamma,
                fusion_alpha=fusion_alpha,
                enable_supervised_fusion=enable_supervised_fusion,
                feature_selection_ratio=feature_selection_ratio,
            )
        raise FileNotFoundError(f"不支持的数据集格式: {dataset_path}")

    if not pcap_dir or not os.path.isdir(pcap_dir):
        raise FileNotFoundError(f"目录不存在: {pcap_dir or split_dir}")

    feature_dir = os.path.join(results_dir, "features")
    os.makedirs(feature_dir, exist_ok=True)

    feature_csvs: List[str] = []
    if pcap_files:
        for idx, pcap in enumerate(pcap_files, 1):
            base = os.path.splitext(os.path.basename(pcap))[0]
            csv_path = os.path.join(feature_dir, f"{base}_features.csv")
            extract_features(pcap, csv_path, progress_cb=None)
            feature_csvs.append(csv_path)
            if progress_cb:
                progress_cb(min(60, int(60 * idx / len(pcap_files))))
    else:
        if progress_cb:
            def _extract_progress(pct: int) -> None:
                pct = max(0, min(100, pct))
                progress_cb(min(60, int(pct * 0.6)))
        else:
            _extract_progress = None

        feature_csvs = extract_features_dir(
            pcap_dir,
            feature_dir,
            workers=workers,
            progress_cb=_extract_progress,
        )

    if progress_cb:
        progress_cb(70)

    dfs = _load_feature_frames(feature_csvs, workers)

    if not dfs:
        raise RuntimeError("未能读取到任何特征 CSV。")

    full_df = pd.concat(dfs, ignore_index=True)

    return _train_from_dataframe(
        full_df,
        results_dir=results_dir,
        models_dir=models_dir,
        contamination=contamination,
        base_estimators=base_estimators,
        progress_cb=progress_cb,
        group_column="pcap_file" if "pcap_file" in full_df.columns else None,
        feature_columns_hint=None,
        rbf_components=rbf_components,
        rbf_gamma=rbf_gamma,
        fusion_alpha=fusion_alpha,
        enable_supervised_fusion=enable_supervised_fusion,
        feature_selection_ratio=feature_selection_ratio,
    )