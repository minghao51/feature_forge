"""Shared helpers for Feature Forge notebooks.

Centralises data loading, LLM client creation, and display utilities
so notebooks stay focused on demos rather than boilerplate.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

try:
    import seaborn as sns
except ImportError:
    sns = None

warnings.filterwarnings("ignore")
os.environ.setdefault("FF_LOG_LEVEL", "warning")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def make_sample_data(
    n_samples: int = 300,
    n_features: int = 8,
    n_informative: int = 5,
    n_redundant: int = 1,
    random_state: int = 42,
    test_size: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a synthetic classification dataset, split into train/test.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=random_state,
    )
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    target = pd.Series(y, name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        df, target, test_size=test_size, random_state=random_state, stratify=target
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Real datasets
# ---------------------------------------------------------------------------


def load_titanic(
    test_size: float = 0.3, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the Titanic dataset, split into train/test.

    Returns (X_train, X_test, y_train, y_test) with sensible feature columns.
    """
    if sns is None:
        raise ImportError("seaborn is required: uv add seaborn")
    df = sns.load_dataset("titanic")

    # Select useful features, drop redundant/leaky columns
    drop_cols = ["survived", "alive", "class", "who", "adult_male", "deck", "embark_town"]
    target = df["survived"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Simple imputation
    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df["fare"] = df["fare"].fillna(df["fare"].median())

    # Encode categoricals
    df["sex"] = df["sex"].map({"male": 0, "female": 1}).astype(int)
    df["alone"] = df["alone"].astype(int)
    embarked_dummies = pd.get_dummies(df["embarked"], prefix="embarked", dtype=int)
    df = pd.concat([df.drop(columns="embarked"), embarked_dummies], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        df, target, test_size=test_size, random_state=random_state, stratify=target
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def get_llm_client(silent: bool = False) -> Any | None:
    """Try to create an LLM client from env vars.

    Returns None (and prints a message) when no API key is configured.
    """
    try:
        from feature_forge.config import Settings
        from feature_forge.llm.factory import create_llm_client

        settings = Settings()
        if (
            settings.llm.api_key is None
            and not os.environ.get("DEEPSEEK_API_KEY")
            and not os.environ.get("OPENAI_API_KEY")
        ):
            if not silent:
                print(
                    "⚠️  No LLM API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY to run LLM-dependent cells."
                )
            return None
        return create_llm_client(settings.llm, retry_config=settings.retry)
    except Exception as exc:
        if not silent:
            print(f"⚠️  Could not create LLM client: {exc}")
        return None


def llm_available() -> bool:
    """Quick check: is an LLM client available?"""
    return get_llm_client(silent=True) is not None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def heading(text: str, level: int = 2) -> None:
    """Print a markdown-style heading for console/notebook display."""
    prefixes = {1: "#", 2: "##", 3: "###", 4: "####"}
    prefix = prefixes.get(level, "##")
    print(f"\n{prefix} {text}\n")


def show_dataframe(df: pd.DataFrame, max_rows: int = 6, title: str | None = None) -> None:
    """Display a DataFrame with an optional title."""
    if title:
        print(f"\n**{title}**\n")
    print(df.head(max_rows).to_string())
    print(f"\nShape: {df.shape}")
