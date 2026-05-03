"""Experiment matrix for defining Cartesian product of experiment parameters."""

from __future__ import annotations

import itertools
from typing import Any


class ExperimentMatrix:
    """Define a Cartesian product of experiment parameters.

    Usage:
        matrix = (
            ExperimentMatrix()
            .datasets(["titanic", "house-prices"])
            .methods({"malmas": ["full"], "openfe": ["openfe"]})
            .seeds([0, 1, 2])
            .models(["xgboost", "lightgbm"])
            .rounds([1, 2, 4])
        )
        configs = matrix.generate()
    """

    def __init__(self) -> None:
        self._params: dict[str, list[Any]] = {}

    def datasets(self, datasets: list[str]) -> ExperimentMatrix:
        self._params["dataset"] = datasets
        return self

    def methods(self, methods: dict[str, list[str]]) -> ExperimentMatrix:
        self._params["method"] = list(methods.keys())
        self._params["method_config"] = [methods]
        return self

    def seeds(self, seeds: list[int]) -> ExperimentMatrix:
        self._params["seed"] = seeds
        return self

    def models(self, models: list[str]) -> ExperimentMatrix:
        self._params["model"] = models
        return self

    def rounds(self, rounds: list[int]) -> ExperimentMatrix:
        self._params["n_rounds"] = rounds
        return self

    def add_param(self, name: str, values: list[Any]) -> ExperimentMatrix:
        self._params[name] = values
        return self

    def generate(self) -> list[dict[str, Any]]:
        """Generate all combinations as dictionaries."""
        if not self._params:
            return []
        keys = list(self._params.keys())
        values = [self._params[k] for k in keys]
        combinations = itertools.product(*values)
        return [dict(zip(keys, combo, strict=False)) for combo in combinations]

    def __len__(self) -> int:
        if not self._params:
            return 0
        total = 1
        for values in self._params.values():
            total *= len(values)
        return total
