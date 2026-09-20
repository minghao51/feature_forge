"""Discovery/evaluation partitioning for leakage-safe feature evaluation.

Feature generation and selection should not see the rows that produce the
reported cross-validation score, otherwise reported gains are optimistically
biased. This module carves a deterministic **discovery** subset (seen by the
method's internal selection) and a disjoint **evaluation** subset (used only
for the reported CV score).

Under the holdout protocol (ADR 0018) partitioning is **fail-closed**: when
either partition cannot independently host the requested number of CV folds,
:data:`resolve_partition` raises :class:`DatasetError` instead of silently
falling back to all-row discovery. Only ``fraction == 0`` (the explicit
compatibility profile) keeps every row in discovery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from feature_forge.exceptions import DatasetError


@dataclass(frozen=True)
class Partition:
    """Row-index partition into discovery and evaluation subsets.

    ``discovery_idx`` / ``evaluation_idx`` are positional integer arrays into
    the original (unshuffled) frame. Either may be empty when the holdout is
    disabled (``fraction == 0``, the compatibility profile per ADR 0018),
    which keeps every row in discovery; otherwise both partitions are
    non-empty and each hosts at least ``cv_folds`` rows (fail closed).
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
    """Resolve a discovery/evaluation row partition, failing closed.

    * ``fraction == 0`` disables the holdout: every row goes to discovery,
      none to evaluation (bit-exact legacy compatibility-profile behavior,
      ADR 0018).
    * Otherwise, when either partition would host fewer than ``cv_folds``
      rows, a :class:`DatasetError` is raised with the partition sizes and
      remediation guidance. The holdout protocol never silently degrades to
      all-row discovery; callers that want that behavior must explicitly
      select the compatibility protocol (``fraction == 0``).

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

    Raises:
        DatasetError: When either partition cannot independently host
            ``cv_folds`` rows, or when the scikit-learn splitter rejects the
            split (e.g. degenerate stratification).
    """
    if fraction <= 0.0:
        return Partition(
            discovery_idx=np.arange(n_rows, dtype=np.int64),
            evaluation_idx=np.empty(0, dtype=np.int64),
        )
    y_values: np.ndarray | None = None
    if stratified:
        if y_for_stratify is None:
            raise ValueError("stratified split requires y_for_stratify")
        y_values = y_for_stratify.to_numpy()
    try:
        if stratified:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
            discovery_idx, evaluation_idx = next(splitter.split(np.arange(n_rows), y_values))
        else:
            splitter = ShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
            discovery_idx, evaluation_idx = next(splitter.split(np.arange(n_rows)))
    except ValueError as exc:
        raise DatasetError(
            f"Cannot create a {fraction:g} discovery/evaluation holdout on "
            f"{n_rows} rows with cv_folds={cv_folds}: {exc}. Lower cv_folds "
            "or the holdout fraction, add rows, or explicitly select the "
            "compatibility protocol for legacy all-row evaluation "
            "(ADR 0018)."
        ) from exc
    discovery_idx = np.asarray(discovery_idx, dtype=np.int64)
    evaluation_idx = np.asarray(evaluation_idx, dtype=np.int64)
    if len(discovery_idx) < cv_folds or len(evaluation_idx) < cv_folds:
        raise DatasetError(
            f"Cannot host {cv_folds} CV folds independently in both "
            f"partitions: {n_rows} rows with holdout fraction {fraction:g} "
            f"yield {len(discovery_idx)} discovery and "
            f"{len(evaluation_idx)} evaluation rows. Lower cv_folds or the "
            "holdout fraction, add rows, or explicitly select the "
            "compatibility protocol for legacy all-row evaluation "
            "(ADR 0018)."
        )
    return Partition(
        discovery_idx=discovery_idx,
        evaluation_idx=evaluation_idx,
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
    Under the holdout protocol an undersized dataset raises
    :class:`DatasetError` (fail closed, ADR 0018); only ``fraction == 0``
    (the compatibility profile) labels every row ``discovery``.

    Raises:
        DatasetError: Under the same conditions as
            :func:`resolve_partition`.
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
