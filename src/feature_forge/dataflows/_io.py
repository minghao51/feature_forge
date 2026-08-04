"""Shared I/O helpers for the medallion dataflow loaders."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from feature_forge.exceptions import DatasetError


@contextmanager
def package_load_guard(layer_name: str) -> Iterator[None]:
    """Wrap a layer package's deserialization phase.

    Any exception raised inside the ``with`` block is re-raised as a
    :class:`~feature_forge.exceptions.DatasetError` with a uniform message,
    unless it is already a ``DatasetError`` (preserving the original, more
    specific context). Semantic-verification phases that run their own checks
    should stay outside this block.
    """
    try:
        yield
    except DatasetError:
        raise
    except Exception as exc:  # loader surface must normalize any deserialization error
        raise DatasetError(f"Unable to load verified {layer_name} package: {exc}") from exc
