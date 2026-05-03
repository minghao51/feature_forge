"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_config():
    """Return a minimal valid configuration dict."""
    return {
        "task": "classification",
        "metric": "auc",
        "n_rounds": 2,
        "random_state": 42,
    }
