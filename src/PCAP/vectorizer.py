"""Utilities for transforming extracted PCAP flow features into ML-ready arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .static_features import extract_pcap_features


@dataclass
class VectorizationResult:
    """Container describing the output of flow vectorization."""

    matrix: np.ndarray
    labels: Optional[np.ndarray]
    feature_names: List[str]

    @property
    def flow_count(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.matrix.shape[1])


@dataclass
class NPZDatasetSummary:
    """Summary describing a saved numpy dataset produced from PCAP flows."""

    path: Path
    flow_count: int
    feature_count: int
    has_labels: bool


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def infer_feature_names(flows: Iterable[Dict[str, object]]) -> List[str]:
    """Infer a stable ordered list of numeric feature names."""

    ordered: Dict[str, None] = {}
    for flow in flows:
        for key, value in flow.items():
            if key == "Label":
                continue
            if _is_numeric(value) and key not in ordered:
                ordered[key] = None
    return list(ordered.keys())


def vectorize_flows(
    flows: Iterable[Dict[str, object]],
    *,
    feature_names: Optional[Sequence[str]] = None,
    default_label: Optional[int] = None,
    dtype: np.dtype = np.float32,
    include_labels: bool = True,
) -> VectorizationResult:
    """Convert extracted flow dictionaries into a numeric matrix.

    Parameters
    ----------
    flows:
        Iterable of flow dictionaries as returned by :func:`extract_pcap_features`.
    feature_names:
        Optional ordered feature names.  When omitted the feature list is inferred
        from the provided flows.
    default_label:
        Optional integer label applied when a flow does not contain a ``Label``
        entry.  When ``None`` missing labels are left unspecified and the returned
        ``labels`` field will be ``None``.
    dtype:
        Desired floating point dtype for the resulting matrix.
    include_labels:
        When ``False`` label processing is skipped even if present within the
        flow dictionaries.  This is useful for inference scenarios where the
        original ``Label`` values should be ignored.
    """

    flow_list = [dict(flow) for flow in flows]
    if not flow_list:
        names = list(feature_names) if feature_names is not None else []
        empty = np.zeros((0, len(names)), dtype=dtype)
        return VectorizationResult(matrix=empty, labels=None, feature_names=names)

    if feature_names is None:
        feature_names = infer_feature_names(flow_list)
    else:
        feature_names = list(feature_names)

    matrix = np.zeros((len(flow_list), len(feature_names)), dtype=dtype)
    label_values: List[int] = []
    has_missing_labels = not include_labels

    for row_index, flow in enumerate(flow_list):
        if include_labels:
            label = flow.get("Label")
            if label is None and default_label is not None:
                label = default_label
                flow["Label"] = default_label

            if label is None or not _is_numeric(label):
                has_missing_labels = True
            else:
                label_values.append(int(label))

        for col_index, name in enumerate(feature_names):
            value = flow.get(name, 0.0)
            matrix[row_index, col_index] = float(value) if _is_numeric(value) else 0.0

    if has_missing_labels:
        labels: Optional[np.ndarray] = None
    else:
        labels = np.asarray(label_values, dtype=np.int64)
        if labels.size != matrix.shape[0]:
            # Some flows might have had non-numeric labels; treat as missing.
            labels = None

    return VectorizationResult(matrix=matrix, labels=labels, feature_names=feature_names)


def vectorize_pcaps(
    inputs: Sequence[Tuple[Union[str, Path], Optional[int]]],
    output_path: Union[str, Path],
    *,
    dtype: np.dtype = np.float32,
) -> NPZDatasetSummary:
    """Extract and vectorize flows from multiple PCAP files.

    Parameters
    ----------
    inputs:
        Sequence of ``(path, label)`` tuples.  The label may be ``None`` for
        unlabeled data.
    output_path:
        Destination ``.npz`` file that will contain ``X``, ``y`` and
        ``feature_names`` arrays.
    dtype:
        Floating point dtype for the feature matrix.
    """

    all_flows: List[Dict[str, object]] = []
    include_labels = any(label is not None for _, label in inputs)

    for path_like, label in inputs:
        path = Path(path_like)
        result = extract_pcap_features(path)
        if not result.get("success", False):
            raise RuntimeError(f"Failed to extract features from {path}: {result.get('error', 'unknown error')}")
        for flow in result.get("flows", []):
            flow_copy = dict(flow)
            if label is not None:
                flow_copy["Label"] = label
            elif not include_labels and "Label" in flow_copy:
                flow_copy.pop("Label")
            all_flows.append(flow_copy)

    vectorized = vectorize_flows(all_flows, dtype=dtype, include_labels=include_labels)
    labels_array = (
        vectorized.labels if vectorized.labels is not None else np.empty((0,), dtype=np.int64)
    )

    np.savez_compressed(
        output_path,
        X=vectorized.matrix,
        y=labels_array,
        feature_names=np.asarray(vectorized.feature_names, dtype=object),
    )

    return NPZDatasetSummary(
        path=Path(output_path),
        flow_count=vectorized.flow_count,
        feature_count=vectorized.feature_count,
        has_labels=labels_array.size > 0,
    )


def load_vectorized_dataset(path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Load a dataset created by :func:`vectorize_pcaps`."""

    with np.load(path, allow_pickle=True) as data:
        matrix = data["X"]
        labels = data["y"]
        feature_names = data["feature_names"].tolist()

    if labels.size == 0:
        return matrix, None, feature_names

    return matrix, labels.astype(np.int64), feature_names
