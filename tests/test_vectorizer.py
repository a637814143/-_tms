from pathlib import Path

import pandas as pd

from src.functions.modeling import _infer_supervised_feature_columns


def test_feature_inference_and_matrix_shape():
    data_path = Path(__file__).parent / "data" / "sample_training.csv"
    df = pd.read_csv(data_path)

    feature_columns = _infer_supervised_feature_columns(df.copy(), "LabelBinary")
    feature_df = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    matrix = feature_df.to_numpy()

    assert matrix.shape[0] == len(df)
    assert matrix.shape[1] == len(feature_columns)
    assert not feature_df.isna().any().any()
