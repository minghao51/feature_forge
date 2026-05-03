"""Shared type aliases and type variables.

These types are used across the codebase to ensure consistency.
"""

from __future__ import annotations

from typing import Any, NewType, TypeVar

import pandas as pd

# Primitive newtypes for domain-specific strings
AgentName = NewType("AgentName", str)
DatasetName = NewType("DatasetName", str)
MetricName = NewType("MetricName", str)
PromptName = NewType("PromptName", str)

# Numeric identifiers
Seed = NewType("Seed", int)
RoundNumber = NewType("RoundNumber", int)

# Core data structures
FeatureSpec = dict[str, Any]
"""Feature specification dict with keys like 'name', 'code', 'logic', 'gain'."""

MemoryEntry = dict[str, Any]
"""Single memory entry dict."""

# Type variables for generic use
T = TypeVar("T")
XType = TypeVar("XType", bound=pd.DataFrame)
YType = TypeVar("YType", bound=pd.Series)

# Task and metric literals
TaskType = str  # "classification" | "regression"
MetricType = str  # "auc" | "acc" | "f1" | "rmse" | "mae" | "r2"

# Router strategy literal
RouterStrategy = str  # "data_driven" | "performance_driven" | "hybrid" | "llm"

# Tracker backend literal
TrackerBackend = str  # "wandb" | "mlflow" | "none"

# LLM provider literal
LLMProvider = str  # "openai" | "deepseek" | "anthropic"
