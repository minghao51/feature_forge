"""Row-local scope contract for generated transformations (plan 23 §4.2, ADR 0018).

A generated candidate is row-local when each output row's value depends only on
that input row. Deterministic row-subset and row-permutation metamorphic probes
verify the property before Gold accepts a candidate under the ``holdout``
protocol; candidates whose values depend on companion rows (whole-frame
statistics such as ``df['a'].mean()``, ``.rank()``, ``.cumsum()``) or on row
position (``df.index``) cannot be verified as row-local and are rejected.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from feature_forge.evaluation.sandbox import SandboxedExecutor

SCOPE_PROBE_SEED = 20260915
"""Deterministic seed for the row-permutation metamorphic probe."""

_PROBE_SOURCE = "gold_scope_probe"
_ROW_SUBSET_MIN_ROWS = 4
_ROW_PERMUTATION_MIN_ROWS = 3
_ROW_SUBSET_REASON = "row-subset probe mismatch"
_ROW_PERMUTATION_REASON = "row-permutation probe mismatch"


def _values_match(actual: pd.Series, expected: pd.Series, *, rtol: float, atol: float) -> bool:
    """Compare positionally aligned probe values against expected full-frame values."""
    actual = actual.reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    if len(actual) != len(expected):
        return False
    if pd.api.types.is_numeric_dtype(actual) and pd.api.types.is_numeric_dtype(expected):
        try:
            return bool(
                np.isclose(
                    actual.to_numpy(dtype=float),
                    expected.to_numpy(dtype=float),
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                ).all()
            )
        except (TypeError, ValueError):
            pass
    return bool(actual.astype(object).eq(expected.astype(object)).fillna(False).all())


def row_local_violations(
    sandbox: SandboxedExecutor,
    code: str,
    frame: pd.DataFrame,
    full_output: pd.DataFrame,
    names: Sequence[str],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> dict[str, str]:
    """Probe generated code for row-locality; return ``{name: reason}`` for violators.

    Runs two deterministic metamorphic probes against the batch's full-frame
    output ``full_output`` (row i of ``full_output`` corresponds to row i of
    ``frame``):

    * row-subset probe (only when ``len(frame) >= 4``): execute the code on
      ``frame.iloc[positions].reset_index(drop=True)`` with
      ``positions = range(0, len(frame), 2)``;
    * row-permutation probe (only when ``len(frame) >= 3``): execute the code on
      ``frame.iloc[perm].reset_index(drop=True)`` with
      ``perm = np.random.default_rng(SCOPE_PROBE_SEED).permutation(len(frame))``.

    For every probe, probe-output row j corresponds to original row positions[j]
    (or perm[j]); a candidate name violates when its probe values are not
    numerically/elementwise equal to the full-frame values for those rows, when
    the name is missing from a probe output, or when a probe execution raises.
    Reasons map to short human-readable strings, e.g. 'row-subset probe mismatch',
    'row-permutation probe mismatch', 'missing from row-subset probe output',
    'scope probe did not execute: <ErrorType>'. Comparison: numeric columns via
    ``np.isclose(..., rtol=rtol, atol=atol)`` elementwise all-close; non-numeric
    via elementwise equality. Deterministic (fixed seed, no randomness from
    callers). ``full_output`` must contain every name in ``names``. Fail closed:
    unverifiable candidates are reported as violating.
    """
    probes: list[tuple[str, pd.DataFrame, NDArray[np.intp], str]] = []
    if len(frame) >= _ROW_SUBSET_MIN_ROWS:
        positions = np.arange(0, len(frame), 2, dtype=np.intp)
        probes.append(
            (
                "row-subset",
                frame.iloc[positions].reset_index(drop=True),
                positions,
                _ROW_SUBSET_REASON,
            )
        )
    if len(frame) >= _ROW_PERMUTATION_MIN_ROWS:
        positions = np.random.default_rng(SCOPE_PROBE_SEED).permutation(len(frame)).astype(np.intp)
        probes.append(
            (
                "row-permutation",
                frame.iloc[positions].reset_index(drop=True),
                positions,
                _ROW_PERMUTATION_REASON,
            )
        )
    violations: dict[str, str] = {}
    for probe_name, probe_frame, positions, mismatch_reason in probes:
        try:
            probe_output = sandbox.execute(code, probe_frame, source=_PROBE_SOURCE)
            if not isinstance(probe_output, pd.DataFrame):
                raise TypeError("scope probe did not return a DataFrame")
        except Exception as exc:
            for name in names:
                violations.setdefault(name, f"scope probe did not execute: {type(exc).__name__}")
            continue
        for name in names:
            if name not in probe_output.columns:
                violations.setdefault(name, f"missing from {probe_name} probe output")
                continue
            try:
                matches = _values_match(
                    probe_output[name],
                    full_output[name].iloc[positions],
                    rtol=rtol,
                    atol=atol,
                )
            except (TypeError, ValueError):
                # Ambiguous shapes (e.g. duplicated column labels) cannot be
                # verified; fail closed and reject the candidate.
                violations.setdefault(name, f"{probe_name} probe comparison failed")
                continue
            if not matches:
                violations.setdefault(name, mismatch_reason)
    return violations
