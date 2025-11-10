"""CLI and optional REST services for malware detector workflows."""

from __future__ import annotations

import argparse
import json
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

logger = get_logger(__name__)

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
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(matrix)
                scores = proba.max(axis=1)
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(matrix)
                if np.ndim(decision) == 1:
                    scores = 1.0 / (1.0 + np.exp(-decision))
                else:
                    scores = decision.max(axis=1)
            else:
                scores = model.predict(matrix)

            preds = model.predict(matrix)
            label_mapping = pipeline.get("label_mapping")
            labels, anomaly_count, normal_count, status_text = summarize_prediction_labels(
                preds,
                label_mapping if isinstance(label_mapping, dict) else None,
            )

            output_df = df.copy()
            output_df["prediction"] = preds
            output_df["prediction_label"] = labels
            output_df["malicious_score"] = scores
            if status_text is not None:
                output_df["prediction_status"] = [
                    label if label in {"异常", "正常"} else status_text for label in labels
                ]

            if output_path is None:
                base = Path(feature_csv).with_suffix("")
                output_path = str(base) + "_predictions.csv"
            output_df.to_csv(output_path, index=False, encoding="utf-8")
            result_info = {
                "output_path": output_path,
                "status_text": status_text,
                "anomaly_count": anomaly_count,
                "normal_count": normal_count,
            }
        else:
            metadata: Dict[str, object] = {}
            if metadata_path and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                    if isinstance(payload, dict):
                        metadata = payload
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning("Failed to read metadata %s: %s", metadata_path, exc)

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
            output_df["is_malicious"] = (preds == -1).astype(int)

            if output_path is None:
                base = Path(feature_csv).with_suffix("")
                output_path = str(base) + "_predictions.csv"
            output_df.to_csv(output_path, index=False, encoding="utf-8")

            anomaly_count = int((preds == -1).sum())
            normal_count = int(len(preds) - anomaly_count)
            status_text = "异常" if anomaly_count > 0 else ("正常" if len(preds) > 0 else None)
            result_info = {
                "output_path": output_path,
                "status_text": status_text,
                "anomaly_count": anomaly_count,
                "normal_count": normal_count,
            }

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
    result = train_unsupervised_on_split(
        args.split_dir,
        args.results_dir,
        args.models_dir,
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
                result = train_unsupervised_on_split(
                    req.split_dir,
                    req.results_dir,
                    req.models_dir,
                )
                log_model_event(
                    "rest.train",
                    {
                        "split_dir": req.split_dir,
                        "results_dir": req.results_dir,
                        "models_dir": req.models_dir,
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