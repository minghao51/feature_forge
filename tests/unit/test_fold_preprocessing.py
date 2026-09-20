"""Fold-local preprocessing unit coverage (plan 23 §4.3, ADR 0018 decision 4)."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from feature_forge.evaluation.preprocessing import (
    UNKNOWN_CATEGORY_SENTINEL,
    FoldPreprocessor,
)


def test_numeric_median_fitted_on_train_rows_only() -> None:
    """Validation NaNs impute with the train-fold median, not their own."""
    fit_frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    preprocessor = FoldPreprocessor.fit(fit_frame)
    assert preprocessor.numeric_medians["a"] == 2.5

    valid_frame = pd.DataFrame({"a": [100.0, 200.0, float("nan")]})
    result = preprocessor.transform(valid_frame)
    assert result["a"].tolist() == [100.0, 200.0, 2.5]


def test_unseen_category_maps_to_sentinel() -> None:
    """Unseen validation categories take the stable -1 sentinel."""
    fit_frame = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
    preprocessor = FoldPreprocessor.fit(fit_frame)

    valid_frame = pd.DataFrame({"cat": ["a", "zzz"]})
    result = preprocessor.transform(valid_frame)
    assert result["cat"].tolist() == [0, UNKNOWN_CATEGORY_SENTINEL]


def test_missing_category_maps_to_sentinel() -> None:
    """NaN categorical values in validation rows also encode as -1."""
    fit_frame = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
    preprocessor = FoldPreprocessor.fit(fit_frame)

    valid_frame = pd.DataFrame({"cat": pd.Series([float("nan"), "b"], dtype=object)})
    result = preprocessor.transform(valid_frame)
    assert result["cat"].tolist() == [UNKNOWN_CATEGORY_SENTINEL, 1]


def test_index_and_column_order_preserved_input_unmutated() -> None:
    """Transform output mirrors the input frame layout without mutating it."""
    fit_frame = pd.DataFrame({"a": [1.0, 2.0], "cat": ["x", "y"]})
    preprocessor = FoldPreprocessor.fit(fit_frame)

    valid_frame = pd.DataFrame(
        {"cat": ["x", "new", float("nan")], "a": [float("nan"), 9.0, 7.0]},
        index=[7, 3, 9],
    )
    before = valid_frame.copy(deep=True)
    result = preprocessor.transform(valid_frame)

    assert list(result.columns) == ["cat", "a"]
    assert result.index.tolist() == [7, 3, 9]
    assert result["cat"].tolist() == [0, -1, -1]
    assert result["a"].tolist() == [1.5, 9.0, 7.0]
    pd.testing.assert_frame_equal(valid_frame, before)


def test_all_nan_numeric_train_column_keeps_nan() -> None:
    """All-NaN train columns keep NaN median parity with the legacy helper."""
    fit_frame = pd.DataFrame({"x": [float("nan"), float("nan"), float("nan")]})
    preprocessor = FoldPreprocessor.fit(fit_frame)
    assert math.isnan(preprocessor.numeric_medians["x"])

    valid_frame = pd.DataFrame({"x": [float("nan"), 5.0]})
    result = preprocessor.transform(valid_frame)
    assert math.isnan(result["x"].iloc[0])
    assert result["x"].iloc[1] == 5.0


def test_boolean_column_classifies_as_numeric() -> None:
    """Boolean columns are numeric (legacy parity), median over 0/1 values."""
    fit_frame = pd.DataFrame({"flag": [True, False, True, True]})
    preprocessor = FoldPreprocessor.fit(fit_frame)
    assert preprocessor.numeric_medians["flag"] == 1.0

    valid_frame = pd.DataFrame({"flag": pd.Series([True, False, float("nan")], dtype=object)})
    result = preprocessor.transform(valid_frame)
    assert result["flag"].tolist() == [True, False, 1.0]


def test_specification_round_trips_and_reflects_fit() -> None:
    """specification() is JSON-serializable and records fitted statistics."""
    fit_frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "cat": ["a", "b", "a", "b"],
            "b": [4.0, 3.0, 2.0, 1.0],
        }
    )
    preprocessor = FoldPreprocessor.fit(fit_frame)

    specification: dict[str, Any] = preprocessor.specification()
    restored: dict[str, Any] = json.loads(json.dumps(specification))
    assert restored == specification

    assert specification["schema_version"] == "1"
    assert specification["fit"] == "training_fold_only"
    assert specification["unknown_category_sentinel"] == -1
    assert list(specification["columns"]) == ["a", "cat", "b"]
    assert specification["columns"]["a"] == {
        "kind": "numeric",
        "imputation": "median",
        "value": 2.5,
    }
    assert specification["columns"]["b"]["value"] == 2.5
    assert specification["columns"]["cat"] == {
        "kind": "categorical",
        "encoding": "ordinal_first_appearance",
        "categories": ["a", "b"],
        "unknown": -1,
    }


def test_specification_serializes_datetime_categories() -> None:
    """Datetime-like categories serialize as ISO strings (JSON-safe specs)."""
    fit_frame = pd.DataFrame(
        {
            "when": pd.to_datetime(["2026-09-15", "2026-09-16", "2026-09-15"]),
        }
    )
    preprocessor = FoldPreprocessor.fit(fit_frame)

    specification: dict[str, Any] = preprocessor.specification()
    assert specification["columns"]["when"]["categories"] == [
        "2026-09-15T00:00:00",
        "2026-09-16T00:00:00",
    ]
    assert json.loads(json.dumps(specification)) == specification
