import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.functions.modeling import train_supervised_on_split


def test_train_and_predict_roundtrip(tmp_path):
    data_path = Path(__file__).parent / "data" / "sample_training.csv"
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    shutil.copy2(data_path, split_dir / data_path.name)

    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"

    result = train_supervised_on_split(
        split_dir=split_dir,
        results_dir=results_dir,
        models_dir=models_dir,
        model_tag="test",
    )

    payload = joblib.load(result["model_path"])
    model = payload["model"]
    feature_names = payload["feature_names"]

    df = pd.read_csv(data_path)
    feature_df = df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    predictions = model.predict(feature_df)

    assert len(predictions) == len(df)
    assert set(np.unique(predictions)).issubset({0, 1})
