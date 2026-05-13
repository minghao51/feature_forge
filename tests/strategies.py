"""Shared Hypothesis strategies for property-based testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import strategies as st

from feature_forge.types import FeatureSpec


@st.composite
def numeric_series(draw, size: int, min_value: float = -1000.0, max_value: float = 1000.0):
    return pd.Series(
        draw(
            st.lists(
                st.floats(min_value, max_value, allow_nan=False, allow_infinity=False),
                min_size=size,
                max_size=size,
            )
        )
    )


@st.composite
def pd_dataframes(
    draw, min_rows: int = 2, max_rows: int = 50, min_cols: int = 1, max_cols: int = 8
):
    n_rows = draw(st.integers(min_rows, max_rows))
    n_cols = draw(st.integers(min_cols, max_cols))
    cols: dict[str, list] = {}
    for i in range(n_cols):
        dtype = draw(st.sampled_from(["float", "int", "str"]))
        col_name = f"col_{i}"
        if dtype == "float":
            cols[col_name] = draw(
                st.lists(
                    st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
        elif dtype == "int":
            cols[col_name] = draw(
                st.lists(st.integers(-100, 100), min_size=n_rows, max_size=n_rows)
            )
        else:
            cols[col_name] = draw(
                st.lists(
                    st.text(min_size=1, max_size=5, alphabet="abcdefghijklmnopqrstuvwxyz"),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
    return pd.DataFrame(cols)


@st.composite
def numeric_pd_dataframes(
    draw, min_rows: int = 5, max_rows: int = 100, min_cols: int = 1, max_cols: int = 10
):
    n_rows = draw(st.integers(min_rows, max_rows))
    n_cols = draw(st.integers(min_cols, max_cols))
    cols: dict[str, list] = {}
    for i in range(n_cols):
        col_name = f"feat_{i}"
        cols[col_name] = draw(
            st.lists(
                st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
    return pd.DataFrame(cols)


@st.composite
def binary_classification_data(draw, min_rows: int = 20, max_rows: int = 200, n_features: int = 3):
    n_rows = draw(st.integers(min_rows, max_rows))
    cols: dict[str, list] = {}
    for i in range(n_features):
        cols[f"feat_{i}"] = draw(
            st.lists(
                st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
    X = pd.DataFrame(cols)
    y = np.array(draw(st.lists(st.sampled_from([0, 1]), min_size=n_rows, max_size=n_rows)))
    return X, y


@st.composite
def regression_data(draw, min_rows: int = 20, max_rows: int = 200, n_features: int = 3):
    n_rows = draw(st.integers(min_rows, max_rows))
    cols: dict[str, list] = {}
    for i in range(n_features):
        cols[f"feat_{i}"] = draw(
            st.lists(
                st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
    X = pd.DataFrame(cols)
    y = np.array(
        draw(
            st.lists(
                st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
    )
    return X, y


@st.composite
def markdown_fenced_code(draw):
    code_body = draw(
        st.text(min_size=1, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz0123456789 =+*\n")
    )
    fence_type = draw(st.sampled_from(["python", "bare", "none"]))
    if fence_type == "python":
        return f"```python\n{code_body}\n```"
    elif fence_type == "bare":
        return f"```\n{code_body}\n```"
    else:
        return code_body


@st.composite
def feature_specs(draw):
    name = draw(st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"))
    ty = draw(st.sampled_from(["numerical", "categorical"]))
    transform = draw(
        st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz [].*+-/")
    )
    n_base = draw(st.integers(1, 3))
    base_columns = [f"col_{i}" for i in range(n_base)]
    return FeatureSpec(
        name=name,
        type=ty,
        transform=transform,
        logic=f"Apply {transform} to {base_columns}",
        base_columns=base_columns,
        agent_name="test_agent",
    )


@st.composite
def llm_messages(draw):
    role = draw(st.sampled_from(["system", "user", "assistant"]))
    content = draw(st.text(min_size=1, max_size=200))
    return {"role": role, "content": content}


@st.composite
def llm_message_lists(draw, min_size: int = 1, max_size: int = 5):
    return draw(st.lists(llm_messages(), min_size=min_size, max_size=max_size))


@st.composite
def valid_metrics(draw):
    return draw(st.sampled_from(["auc", "acc", "f1", "rmse", "mae", "r2", "nrmse"]))


@st.composite
def numpy_y_true_y_pred(draw, n: int = 100, task: str = "classification"):
    if task == "classification":
        y_true = np.array(draw(st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n)))
        y_pred = np.array(
            draw(
                st.lists(
                    st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
                    min_size=n,
                    max_size=n,
                )
            )
        )
    else:
        y_true = np.array(
            draw(
                st.lists(
                    st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
                    min_size=n,
                    max_size=n,
                )
            )
        )
        y_pred = np.array(
            draw(
                st.lists(
                    st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
                    min_size=n,
                    max_size=n,
                )
            )
        )
    return y_true, y_pred
