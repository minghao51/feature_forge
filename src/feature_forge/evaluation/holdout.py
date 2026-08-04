"""Discovery/evaluation partitioning for leakage-safe feature evaluation.

Feature generation and selection should not see the rows that produce the
reported cross-validation score, otherwise reported gains are optimistically
biased. This module carves a deterministic **discovery** subset (seen by the
method's internal selection) and a disjoint **evaluation** subset (used only
for the reported CV score).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


@dataclass(frozen=True)
class Partition:
    """Row-index partition into discovery and evaluation subsets.

    ``discovery_idx`` / ``evaluation_idx`` are positional integer arrays into
    the original (unshuffled) frame. Either may be empty when the holdout is
    disabled (``fraction == 0``) or when a partition is too small to host the
    requested number of CV folds (see :func:`resolve_partition`).
    """

    discovery_idx: np.ndarray
    evaluation_idx: np.ndarray

    @property
    def holdout_active(self) -> bool:
        """True when an evaluation partition is actually reserved."""
        return len(self.evaluation_idx) > 0 and len(self.discovery_idx) > 0


def resolve_partition(
    n_rows: int,
    *,
    fraction: float,
    cv_folds: int,
    stratified: bool,
    seed: int,
    y_for_stratify: pd.Series | None = None,
) -> Partition:
    """Resolve a discovery/evaluation row partition, degrading gracefully.

    * ``fraction == 0`` disables the holdout: every row goes to discovery,
      none to evaluation (bit-exact legacy behavior).
    * When either partition would have fewer than ``cv_folds`` rows, the
      holdout is dropped (every row → discovery) and a warning is left to
      the caller via :attr:`Partition.holdout_active`.

    Args:
        n_rows: Total number of rows to partition.
        fraction: Evaluation-holdout fraction in ``[0.0, 0.5)``. See
            :class:`~feature_forge.config.EvaluationConfig`.
        cv_folds: CV fold count that each partition must be able to host.
        stratified: Stratify the split on ``y_for_stratify`` (classification).
        seed: Deterministic split seed.
        y_for_stratify: Target aligned with the rows; required when
            ``stratified`` is True.

    Returns:
        A :class:`Partition` of positional indices into the input rows.
    """
    if fraction <= 0.0:
        return Partition(
            discovery_idx=np.arange(n_rows, dtype=np.int64),
            evaluation_idx=np.empty(0, dtype=np.int64),
        )
    if n_rows < 2 * cv_folds:
        # Both partitions must host `cv_folds` validation rows each; if the
        # data is too small, fall back to whole-data evaluation rather than
        # producing a degenerate split.
        return Partition(
            discovery_idx=np.arange(n_rows, dtype=np.int64),
            evaluation_idx=np.empty(0, dtype=np.int64),
        )
    if stratified:
        if y_for_stratify is None:
            raise ValueError("stratified split requires y_for_stratify")
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        discovery_idx, evaluation_idx = next(
            splitter.split(np.arange(n_rows), y_for_stratify.to_numpy())
        )
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        discovery_idx, evaluation_idx = next(splitter.split(np.arange(n_rows)))
    return Partition(
        discovery_idx=np.asarray(discovery_idx, dtype=np.int64),
        evaluation_idx=np.asarray(evaluation_idx, dtype=np.int64),
    )


def assign_row_partitions(
    n_rows: int,
    *,
    fraction: float,
    cv_folds: int,
    stratified: bool,
    seed: int,
    y_for_stratify: pd.Series | None = None,
) -> np.ndarray:
    """Return a per-row ``partition`` label array (``discovery``/``evaluation``).

    Mirrors :func:`resolve_partition` but returns a string label per row, used
    to build the Silver ``fold_assignments.partition`` column. The label array
    has length ``n_rows`` and every row is assigned (no row left unassigned).
    """
    partition = resolve_partition(
        n_rows,
        fraction=fraction,
        cv_folds=cv_folds,
        stratified=stratified,
        seed=seed,
        y_for_stratify=y_for_stratify,
    )
    labels = np.empty(n_rows, dtype=object)
    labels[:] = "discovery"
    if partition.holdout_active:
        labels[partition.evaluation_idx] = "evaluation"
    return labels
