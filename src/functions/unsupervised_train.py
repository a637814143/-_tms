"""无监督异常检测训练流程。"""

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from joblib import dump

from src.functions.feature_extractor import extract_features, extract_features_dir
from src.functions.anomaly_detector import EnsembleAnomalyDetector

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


def _is_npz(path: str) -> bool:
    return path.lower().endswith(".npz")


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
    """尝试把标签列解析为 0/1。"""

    if series.empty:
        return None

    if series.dtype == bool:
        return series.fillna(False).astype(int).to_numpy()

    try:
        numeric = pd.to_numeric(series, errors="coerce")
    except Exception:
        numeric = None

    if numeric is not None and not numeric.isna().all():
        arr = numeric.fillna(0.0).to_numpy()
        binary = np.where(arr > 0, 1, 0).astype(int)
        if np.unique(binary).size >= 2:
            return binary

    normalized = (
        series.fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if normalized.eq("unknown").all():
        return None

    parsed = np.full(len(normalized), fill_value=-1, dtype=int)

    for idx, value in enumerate(normalized):
        if value in NORMAL_TOKENS:
            parsed[idx] = 0
        elif value in ANOMALY_TOKENS:
            parsed[idx] = 1
        else:
            try:
                numeric_value = float(value)
            except Exception:
                numeric_value = None
            if numeric_value is not None:
                parsed[idx] = 1 if numeric_value > 0 else 0

    unresolved_mask = parsed == -1
    if np.all(unresolved_mask):
        # 完全未知的取值，尝试根据主频词推断
        value_counts = normalized.value_counts()
        if value_counts.size < 2:
            return None
        majority = value_counts.idxmax()
        majority_mask = normalized.eq(majority).to_numpy()
        parsed = np.where(majority_mask, 0, 1)
    elif np.any(unresolved_mask):
        unresolved_values = normalized[unresolved_mask]
        known_normal = np.count_nonzero(parsed == 0)
        known_anomaly = np.count_nonzero(parsed == 1)

        unresolved_unique = unresolved_values.unique()
        unresolved_ratio = len(unresolved_values) / len(parsed)

        if known_normal == 0 and known_anomaly == 0:
            # 没有任何已识别标签，仍退化为主频词推断
            value_counts = normalized.value_counts()
            if value_counts.size < 2:
                return None
            majority = value_counts.idxmax()
            majority_mask = normalized.eq(majority).to_numpy()
            parsed = np.where(majority_mask, 0, 1)
        elif known_normal > 0 and known_anomaly == 0:
            # 已知正常类别存在。若未知类别种类丰富，则把它们视为异常；
            # 若仅出现单一取值且占比过大，则认为信息不足直接放弃。
            if unresolved_unique.size == 1 and unresolved_ratio > 0.4:
                return None
            parsed[unresolved_mask] = 1
        elif known_anomaly > 0 and known_normal == 0:
            if unresolved_unique.size == 1 and unresolved_ratio > 0.4:
                return None
            parsed[unresolved_mask] = 0
        else:
            # 已经识别出正负样本，剩余全部按异常处理
            parsed[unresolved_mask] = 1

    unique = np.unique(parsed)
    if unique.size < 2:
        return None
    return parsed.astype(int)


def _extract_ground_truth(df: pd.DataFrame) -> Tuple[Optional[str], Optional[np.ndarray]]:
    for column in df.columns:
        if column.lower() in GROUND_TRUTH_COLUMN_CANDIDATES:
            series = df[column]
            binary = _series_to_binary(series)
            if binary is not None and binary.size == len(df):
                return column, binary
    return None, None


def _latest_preprocessed_csv(directory: str) -> Optional[str]:
    candidates = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if _is_preprocessed_csv(os.path.join(directory, name))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def _manifest_path_for(dataset_path: str) -> str:
    base, _ = os.path.splitext(dataset_path)
    return f"{base}_manifest.csv"


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


def _prepare_feature_matrix(
    df: pd.DataFrame, feature_columns_hint: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """提取用于训练的特征矩阵。

    如果提供了 ``feature_columns_hint``，则严格按照提示顺序筛选列，
    这可以确保预处理阶段与训练阶段使用完全一致的特征集合。
    若部分列缺失，会抛出异常提示用户重新执行预处理流程。
    """

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
            if is_numeric_dtype(df[col]):
                candidate_cols.append(col)

    if not candidate_cols:
        raise RuntimeError("未发现可用于训练的数值特征。")

    numeric_df = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    # 以列中位数进行填充，进一步提升鲁棒性
    medians = numeric_df.median(axis=0, skipna=True)
    medians = medians.fillna(0.0)
    filled = numeric_df.fillna(medians).astype("float32")

    return filled.to_numpy(dtype=float, copy=False), candidate_cols


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
) -> dict:
    if df.empty:
        raise RuntimeError("训练数据为空。")

    working_df = df.copy()
    ground_truth_column, ground_truth = _extract_ground_truth(working_df)
    X, feature_columns = _prepare_feature_matrix(
        working_df, feature_columns_hint=feature_columns_hint
    )

    adaptive_neighbors = int(max(15, min(120, np.sqrt(len(working_df)) * 2)))
    detector = EnsembleAnomalyDetector(
        contamination=contamination,
        n_estimators=max(256, base_estimators * 4),
        n_neighbors=adaptive_neighbors,
        random_state=42,
    )

    pipeline = Pipeline(
        [
            ("variance_filter", VarianceThreshold(threshold=1e-6)),
            ("scaler", StandardScaler()),
            (
                "gaussianizer",
                QuantileTransformer(
                    output_distribution="normal",
                    subsample=200000,
                    random_state=42,
                ),
            ),
            ("detector", detector),
        ]
    )

    if progress_cb:
        progress_cb(70)

    if ground_truth is not None:
        pipeline.fit(X, ground_truth)
    else:
        pipeline.fit(X)
    try:
        pipeline.feature_names_in_ = np.asarray(feature_columns)
    except Exception:
        pass
    detector = pipeline.named_steps["detector"]

    supervised_metrics = None
    if ground_truth is not None:
        transformer = Pipeline(pipeline.steps[:-1])
        transformed = transformer.transform(X)
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)
        supervised_model = HistGradientBoostingClassifier(
            max_depth=7,
            learning_rate=0.1,
            max_iter=400,
            l2_regularization=1.0,
            random_state=42,
        )
        supervised_model.fit(transformed, ground_truth)
        supervised_proba = supervised_model.predict_proba(transformed)[:, 1]
        thr_candidates = np.linspace(0.1, 0.9, 41)
        best_thr = 0.5
        best_f1 = -1.0
        best_prec = 0.0
        best_rec = 0.0
        for thr in thr_candidates:
            preds = (supervised_proba >= thr).astype(int)
            f1 = f1_score(ground_truth, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
                best_prec = precision_score(ground_truth, preds, zero_division=0)
                best_rec = recall_score(ground_truth, preds, zero_division=0)
        detector.supervised_model_ = supervised_model
        detector.supervised_threshold_ = float(best_thr)
        detector.supervised_input_dim_ = transformed.shape[1]
        detector.last_supervised_scores_ = supervised_proba.astype(float)
        supervised_metrics = {
            "precision": float(best_prec),
            "recall": float(best_rec),
            "f1": float(best_f1),
            "threshold": float(best_thr),
        }

    scores = None
    if detector.last_combined_scores_ is not None:
        scores = detector.last_combined_scores_
    elif detector.fit_decision_scores_ is not None:
        scores = detector.fit_decision_scores_
    else:
        scores = detector.score_samples(pipeline.named_steps["scaler"].transform(X))
    scores = np.asarray(scores, dtype=float)

    preds = pipeline.predict(X)
    is_malicious = (preds == -1).astype(int)

    vote_ratio = None
    if detector.last_vote_ratio_ is not None:
        vote_ratio = detector.last_vote_ratio_
    elif detector.fit_votes_:
        vote_ratio = np.vstack([np.where(v == -1, 1.0, 0.0) for v in detector.fit_votes_.values()]).mean(axis=0)
    else:
        vote_ratio = np.ones_like(is_malicious, dtype=float)
    vote_ratio = np.asarray(vote_ratio, dtype=float)

    threshold = detector.threshold_ if detector.threshold_ is not None else float(np.quantile(scores, contamination))
    vote_threshold = detector.vote_threshold_ if detector.vote_threshold_ is not None else float(np.mean(vote_ratio))
    vote_threshold = float(np.clip(vote_threshold, 0.0, 1.0))

    score_std = float(np.std(scores) or 1.0)
    conf_from_score = 1.0 / (1.0 + np.exp((scores - threshold) / (score_std + 1e-6)))
    vote_component = np.clip((vote_ratio - vote_threshold) / max(1e-6, (1.0 - vote_threshold)), 0.0, 1.0)
    anomaly_confidence = np.clip((conf_from_score + vote_component) / 2.0, 0.0, 1.0)
    if detector.last_supervised_scores_ is not None:
        anomaly_confidence = detector.last_supervised_scores_.astype(float)
    elif detector.last_calibrated_scores_ is not None:
        anomaly_confidence = detector.last_calibrated_scores_.astype(float)

    working_df["anomaly_score"] = scores
    working_df["vote_ratio"] = vote_ratio
    working_df["anomaly_confidence"] = anomaly_confidence
    working_df["is_malicious"] = is_malicious

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

    latest_pipeline_path = os.path.join(models_dir, "latest_iforest_pipeline.joblib")
    try:
        shutil.copy2(pipeline_path, latest_pipeline_path)
    except Exception:
        dump(pipeline, latest_pipeline_path)

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

    metadata = {
        "timestamp": timestamp,
        "contamination": contamination,
        "base_estimators": base_estimators,
        "feature_columns": feature_columns,
        "threshold": float(threshold),
        "score_std": float(score_std),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "vote_mean": float(np.mean(vote_ratio)),
        "vote_threshold": float(vote_threshold),
        "threshold_breakdown": detector.threshold_breakdown_,
        "detectors": list(detector.detectors_.keys()),
        "estimated_precision": float(anomaly_confidence[is_malicious == 1].mean() if is_malicious.any() else 0.0),
        "estimated_anomaly_ratio": float(is_malicious.mean()),
        "results_csv": results_csv,
        "summary_csv": summary_csv,
        "gaussianizer_path": gaussianizer_path,
    }

    if detector.calibration_report_ is not None:
        metadata["calibration"] = detector.calibration_report_
        metadata["calibration_threshold"] = float(
            detector.calibration_threshold_ if detector.calibration_threshold_ is not None else 0.5
        )

    if supervised_metrics is not None:
        metadata["supervised"] = supervised_metrics

    if ground_truth is not None:
        precision = precision_score(ground_truth, is_malicious, zero_division=0)
        recall = recall_score(ground_truth, is_malicious, zero_division=0)
        f1 = f1_score(ground_truth, is_malicious, zero_division=0)
        acc = accuracy_score(ground_truth, is_malicious)
        metadata["ground_truth_column"] = ground_truth_column
        metadata["ground_truth_ratio"] = float(np.mean(ground_truth))
        metadata["evaluation"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(acc),
        }

    metadata_path = os.path.join(models_dir, f"iforest_metadata_{timestamp}.json")
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    latest_meta_path = os.path.join(models_dir, "latest_iforest_metadata.json")
    with open(latest_meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    if progress_cb:
        progress_cb(100)

    return {
        "results_csv": results_csv,
        "summary_csv": summary_csv,
        "model_path": pipeline_path,
        "scaler_path": scaler_path,
        "pipeline_path": pipeline_path,
        "pipeline_latest": latest_pipeline_path,
        "metadata_path": metadata_path,
        "metadata_latest": latest_meta_path,
        "packets": len(working_df),
        "flows": len(working_df),
        "malicious": int(is_malicious.sum()),
        "contamination": contamination,
        "threshold": float(threshold),
        "vote_threshold": float(vote_threshold),
        "estimated_precision": metadata["estimated_precision"],
        "threshold_breakdown": detector.threshold_breakdown_,
        "feature_columns": feature_columns,
        "gaussianizer_path": gaussianizer_path,
        "ground_truth_column": ground_truth_column,
        "evaluation": metadata.get("evaluation"),
        "supervised": supervised_metrics,
    }


def _train_from_preprocessed_csv(
    dataset_path: str,
    *,
    results_dir: str,
    models_dir: str,
    contamination: float,
    base_estimators: int,
    progress_cb=None,
) -> dict:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"预处理数据集不存在: {dataset_path}")

    df = pd.read_csv(dataset_path, encoding="utf-8")
    if df.empty:
        raise RuntimeError("预处理数据集为空，无法训练。")

    manifest_path = _manifest_path_for(dataset_path)
    manifest_col = None
    feature_columns_hint: Optional[List[str]] = None

    meta_path = dataset_path.replace(".csv", "_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta_payload = json.load(fh)
            if isinstance(meta_payload, dict):
                cols = meta_payload.get("feature_columns")
                if isinstance(cols, list):
                    feature_columns_hint = [str(col) for col in cols]
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
    )


def _train_from_npz(
    dataset_path: str,
    *,
    results_dir: str,
    models_dir: str,
    contamination: float,
    base_estimators: int,
    progress_cb=None,
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

    return _train_from_dataframe(
        df,
        results_dir=results_dir,
        models_dir=models_dir,
        contamination=contamination,
        base_estimators=base_estimators,
        progress_cb=progress_cb,
        group_column=manifest_col,
        feature_columns_hint=list(columns),
    )


def train_unsupervised_on_split(
    split_dir: str,
    results_dir: str,
    models_dir: str,
    contamination: float = 0.05,
    base_estimators: int = 50,
    progress_cb=None,
    workers: int = 4,
):
    """
    无监督训练：
    - 支持直接加载预处理 CSV 或 NPZ 数据集
    - 对 PCAP 目录进行特征提取（多线程）后再训练
    """

    _ensure_dirs(results_dir, models_dir)

    dataset_path: Optional[str] = None
    pcap_dir: Optional[str] = None
    pcap_files: List[str] = []

    if not split_dir:
        raise FileNotFoundError("未提供训练数据路径")

    if os.path.isfile(split_dir):
        if _is_preprocessed_csv(split_dir):
            dataset_path = split_dir
        elif _is_npz(split_dir):
            dataset_path = split_dir
        elif split_dir.lower().endswith((".pcap", ".pcapng")):
            pcap_dir = os.path.dirname(split_dir) or os.path.abspath(os.path.join(split_dir, os.pardir))
            pcap_files = [split_dir]
        else:
            raise FileNotFoundError(f"不支持的训练文件: {split_dir}")
    elif os.path.isdir(split_dir):
        dataset_path = _latest_preprocessed_csv(split_dir)
        if dataset_path is None:
            dataset_path = _latest_npz_in_dir(split_dir)
        if dataset_path is None:
            pcap_dir = split_dir
    else:
        raise FileNotFoundError(f"路径不存在: {split_dir}")

    if dataset_path:
        if dataset_path.lower().endswith(".csv"):
            return _train_from_preprocessed_csv(
                dataset_path,
                results_dir=results_dir,
                models_dir=models_dir,
                contamination=contamination,
                base_estimators=base_estimators,
                progress_cb=progress_cb,
            )
        return _train_from_npz(
            dataset_path,
            results_dir=results_dir,
            models_dir=models_dir,
            contamination=contamination,
            base_estimators=base_estimators,
            progress_cb=progress_cb,
        )

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
    )