"""Fold-local preprocessing fitted on training-fold rows only (ADR 0018 §4).

Numeric columns are median-imputed and non-numeric columns are ordinally
encoded using statistics fitted on a fold's training rows only; validation
rows are transformed with those train-only statistics, and categories never
seen during fitting map to the documented stable sentinel ``-1``
(``UNKNOWN_CATEGORY_SENTINEL``, matching ``pandas.Categorical.codes``
semantics for unseen values). This is the typed small helper allowed by
ADR 0018 decision 4 in place of a second evaluation framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

UNKNOWN_CATEGORY_SENTINEL: int = -1


def _plain_value(value: Any) -> Any:
    """Coerce numpy scalars and datetimes to plain JSON-serializable values."""
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat) and not isinstance(value, (str, bytes)):
        return isoformat()
    return value


@dataclass(frozen=True)
class FoldPreprocessor:
    """Numeric/categorical preprocessing fitted on one fold's training rows.

    Attributes:
        numeric_medians: Train-fold median per numeric column (pandas
            skip-NaN semantics; an all-NaN training column keeps NaN parity
            with the legacy whole-frame helper).
        categorical_categories: Ordered train-fold categories per non-numeric
            column, in first-appearance order (mirrors the legacy
            ``pd.Categorical`` coding for training rows and is deterministic
            for a fixed frame).
    """

    numeric_medians: dict[str, float]
    categorical_categories: dict[str, tuple[Any, ...]]
    _column_order: tuple[str, ...] = ()

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> FoldPreprocessor:
        """Fit imputation/encoding statistics on training rows only."""
        numeric_medians: dict[str, float] = {}
        categorical_categories: dict[str, tuple[Any, ...]] = {}
        column_order: list[str] = []
        for column in frame.columns:
            name = str(column)
            column_order.append(name)
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                numeric_medians[name] = float(series.median())
            else:
                categorical_categories[name] = tuple(series.dropna().unique())
        return cls(
            numeric_medians=numeric_medians,
            categorical_categories=categorical_categories,
            _column_order=tuple(column_order),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform rows with the train-only statistics.

        Unknown categories (and missing categorical values) map to
        ``UNKNOWN_CATEGORY_SENTINEL``; numeric NaNs map to the train-fold
        median. Returns a new frame with the input's index and column order;
        the input is never mutated.
        """
        result = frame.copy(deep=True)
        for column in frame.columns:
            name = str(column)
            if name in self.numeric_medians:
                result[name] = result[name].fillna(self.numeric_medians[name])
            elif name in self.categorical_categories:
                result[name] = pd.Categorical(
                    result[name], categories=self.categorical_categories[name]
                ).codes
        return result

    def specification(self) -> dict[str, Any]:
        """Return the fitted preprocessing specification (provenance).

        The returned mapping is ``json.dumps``-able; column order under
        ``columns`` follows the fit-frame order.
        """
        order = self._column_order or (*self.numeric_medians, *self.categorical_categories)
        columns: dict[str, Any] = {}
        for name in order:
            if name in self.numeric_medians:
                columns[name] = {
                    "kind": "numeric",
                    "imputation": "median",
                    "value": self.numeric_medians[name],
                }
            elif name in self.categorical_categories:
                columns[name] = {
                    "kind": "categorical",
                    "encoding": "ordinal_first_appearance",
                    "categories": [
                        _plain_value(category) for category in self.categorical_categories[name]
                    ],
                    "unknown": UNKNOWN_CATEGORY_SENTINEL,
                }
        return {
            "schema_version": "1",
            "fit": "training_fold_only",
            "unknown_category_sentinel": UNKNOWN_CATEGORY_SENTINEL,
            "columns": columns,
        }
