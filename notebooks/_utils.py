"""Shared helpers for Feature Forge notebooks.

Centralises data loading, LLM client creation, and display utilities
so notebooks stay focused on demos rather than boilerplate.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
os.environ.setdefault("FF_LOG_LEVEL", "warning")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _openml(
    name: str,
    version: int = 1,
) -> pd.DataFrame:
    """Fetch an OpenML dataset as a DataFrame.

    Uses sklearn's cached fetch_openml internally.
    """
    from sklearn.datasets import fetch_openml

    d = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
    df = d.frame
    df.columns = [c.strip() for c in df.columns]
    return df  # type: ignore[no-any-return]


def _split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into train/test, returning (X_train, X_test, y_train, y_test)."""
    y = df[target_col].astype(int) if df[target_col].dtype.name == "category" else df[target_col]
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Banknote Authentication  (Notebook 01)
# ---------------------------------------------------------------------------

BANKNOTE_COLUMNS = ["variance", "skewness", "curtosis", "entropy"]


def load_banknote(
    test_size: float = 0.3, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Banknote authentication (UCI).

    1372 samples, 4 numeric features extracted from wavelet-transformed
    banknote images.  Binary target: 1=forged, 0=genuine.
    """
    df = _openml("banknote-authentication")
    df.columns = [*BANKNOTE_COLUMNS, "is_forged"]
    # Original category values: "1"=forged, "2"=genuine → convert to 1/0
    df["is_forged"] = (df["is_forged"].astype(str) == "1").astype(int)
    return _split(df, "is_forged", test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Indian Liver Patient Dataset  (Notebook 02)
# ---------------------------------------------------------------------------

ILPD_COLUMNS = [
    "age_years",
    "gender",
    "total_bilirubin",
    "direct_bilirubin",
    "alkaline_phosphotase",
    "alamine_aminotransferase",
    "aspartate_aminotransferase",
    "total_proteins",
    "albumin",
    "albumin_globulin_ratio",
]


def load_ilpd(
    test_size: float = 0.3, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Indian Liver Patient Dataset (UCI).

    583 samples, 10 mixed-type features (numerical + categorical).
    Binary target: 1=liver disease, 2=no liver disease (inverted to 0/1).
    """
    df = _openml("ilpd")
    df.columns = [*ILPD_COLUMNS, "has_disease"]
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1}).astype(int)
    df["has_disease"] = (df["has_disease"].astype(str) == "1").astype(int)
    return _split(df, "has_disease", test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Steel Plates Fault  (Notebook 03)
# ---------------------------------------------------------------------------

STEEL_COLUMNS = [
    "x_minimum",
    "x_maximum",
    "y_minimum",
    "y_maximum",
    "pixels_areas",
    "x_perimeter",
    "y_perimeter",
    "sum_luminosity",
    "min_luminosity",
    "max_luminosity",
    "conveyer_length",
    "steel_A300",
    "steel_A400",
    "plate_thickness",
    "edges_index",
    "empty_index",
    "square_index",
    "outside_x_index",
    "edges_index_v2",
    "outside_global_index",
    "log_areas",
    "log_x_index",
    "log_y_index",
    "orientation_index",
    "luminosity_index",
    "sigmoid_areas",
    "pastel_0",
    "pastel_1",
    "pastel_2",
    "pastel_3",
    "pastel_4",
    "pastel_5",
    "pastel_6",
]


def load_steel_plates(
    test_size: float = 0.3, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Steel Plates Fault (UCI).

    1941 samples, 33 numeric features describing steel plate fault
    geometry. Binary target: 1=has fault, 2=no fault (inverted to 0/1).
    """
    df = _openml("steel-plates-fault")
    df.columns = [*STEEL_COLUMNS, "has_fault"]
    df["has_fault"] = (df["has_fault"].astype(str) == "1").astype(int)
    return _split(df, "has_fault", test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Heart Disease (Cleveland)  (Notebook 04)
# ---------------------------------------------------------------------------

HEART_FEATURE_NAMES = {
    "age": "age",
    "sex": "sex",
    "cp": "chest_pain_type",
    "trestbps": "resting_bp",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "restecg": "rest_ecg",
    "thalach": "max_heart_rate",
    "exang": "exercise_angina",
    "oldpeak": "st_depression",
    "slope": "st_slope",
    "ca": "colored_vessels",
    "thal": "thalassemia",
}


def load_heart_disease(
    test_size: float = 0.3, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Heart Disease (Cleveland) from UCI via OpenML.

    303 samples, 13 mixed features (numeric + categorical).
    Binary target: 0=no disease (<50% narrowing), 1=disease (>50%).
    """
    df = _openml("heart-c")
    df = df.rename(columns=HEART_FEATURE_NAMES)
    target_col = "num"
    df = df.rename(columns={target_col: "has_disease"})

    encode_map = {
        "sex": {"male": 0, "female": 1},
        "chest_pain_type": {"typ_angina": 0, "asympt": 1, "non_anginal": 2, "atyp_angina": 3},
        "fasting_blood_sugar": {"f": 0, "t": 1},
        "rest_ecg": {"normal": 0, "st_t_wave_abnormality": 1, "left_vent_hyper": 2},
        "exercise_angina": {"no": 0, "yes": 1},
        "st_slope": {"up": 0, "flat": 1, "down": 2},
        "thalassemia": {"normal": 0, "fixed_defect": 1, "reversable_defect": 2},
    }
    for col, mapping in encode_map.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].map(mapping), errors="coerce").fillna(-1).astype(int)

    df["has_disease"] = (df["has_disease"] == ">50_1").astype(int)
    df = df.fillna(df.median(numeric_only=True))
    return _split(df, "has_disease", test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Phoneme  (Notebook 05)
# ---------------------------------------------------------------------------

PHONEME_COLUMNS = [
    "harmonicity",
    "mid_spectral",
    "high_freq_power",
    "vowel_duration",
    "plosive_amp",
]


def load_phoneme(
    test_size: float = 0.3, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Phoneme dataset from OpenML.

    5404 samples, 5 numeric features extracted from speech recordings.
    Binary target: 0=class 1 (phoneme), 1=class 2 (phoneme).
    """
    df = _openml("phoneme")
    df.columns = [*PHONEME_COLUMNS, "target"]
    df["target"] = (df["target"].astype(str) == "2").astype(int)
    return _split(df, "target", test_size=test_size, random_state=random_state)


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
                    "\u26a0\ufe0f  No LLM API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY to run LLM-dependent cells."
                )
            return None
        return create_llm_client(settings.llm, retry_config=settings.retry)
    except Exception as exc:
        if not silent:
            print(f"\u26a0\ufe0f  Could not create LLM client: {exc}")
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
