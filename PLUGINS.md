# Plugin Developer Guide

Feature Forge supports extensibility via Python entry points. You can register custom baselines, datasets, models, and metrics without modifying the core package.

## Table of Contents

- [Baseline Plugins](#baseline-plugins)
- [Model Plugins](#model-plugins)
- [Metric Plugins](#metric-plugins)
- [Dataset Plugins](#dataset-plugins)
- [Using Plugins with ExperimentalPlatform](#using-plugins-with-experimentalplatform)

## Baseline Plugins

### Contract

A baseline plugin must satisfy `BaselineProtocol` — a `@runtime_checkable` protocol that requires:

- `name: str` — Human-readable name
- `fit(X_train, y_train) -> Self` — Fit the baseline
- `transform(X) -> pd.DataFrame` — Generate features
- `fit_transform(X_train, y_train) -> pd.DataFrame` — Convenience method
- `generated_scripts -> list[str]` — Generated code blocks
- `feature_metadata -> list[dict]` — Feature descriptions
- `get_artifacts() -> dict` — Collected artifacts

You do **not** need to import from `feature_forge` to create a baseline.

### Example

```python
# mymethod/baseline.py
from typing import Any
import pandas as pd


class MyMethod:
    name = "mymethod"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "MyMethod":
        # Learn from training data
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Generate new features
        return pd.DataFrame({"my_feature": X.iloc[:, 0] * 2})

    def fit_transform(self, X_train, y_train) -> pd.DataFrame:
        self.fit(X_train, y_train)
        return self.transform(X_train)

    @property
    def generated_scripts(self) -> list[str]:
        return ["new_col = df['col'] * 2"]

    @property
    def feature_metadata(self) -> list[dict[str, Any]]:
        return [{"name": "my_feature", "type": "numerical"}]

    def get_artifacts(self) -> dict[str, Any]:
        return {}
```

### Registration

In your `pyproject.toml`:

```toml
[project.entry-points."feature_forge.baselines"]
mymethod = "mymethod.baseline:MyMethod"
```

After `pip install`, `MyMethod` is auto-discovered:

```python
from feature_forge import ExperimentalPlatform

platform = ExperimentalPlatform()
"mymethod" in platform.list_baselines()  # True
```

## Model Plugins

### Contract

A model plugin is a factory function with signature:

```python
def create_model(task: str, random_state: int = 42) -> Any:
    """Return a sklearn-compatible estimator."""
```

### Example

```python
# feature_forge_catboost/__init__.py
def create_catboost(task: str, random_state: int = 42):
    from catboost import CatBoostClassifier, CatBoostRegressor
    kwargs = {"iterations": 500, "random_state": random_state, "verbose": False}
    return CatBoostClassifier(**kwargs) if task == "classification" else CatBoostRegressor(**kwargs)
```

### Registration

```toml
[project.entry-points."feature_forge.models"]
catboost = "feature_forge_catboost:create_catboost"
```

## Metric Plugins

### Contract

A metric plugin is a function with signature:

```python
def metric_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return a scalar score. Higher is better."""
```

### Registration

```toml
[project.entry-points."feature_forge.metrics"]
log_loss = "custom_metrics:log_loss_score"
```

## Dataset Plugins

### Contract

A dataset plugin is a loader function with signature:

```python
def dataset_loader() -> dict:
    """Return dict with keys: train, test, target, metadata."""
```

The returned dict should match the format:

```python
{
    "train": pd.DataFrame,      # training data with target column
    "test": pd.DataFrame,       # test data (optional)
    "target": str,              # target column name
    "metadata": dict,           # dataset info (task, source, etc.)
}
```

### Registration

```toml
[project.entry-points."feature_forge.datasets"]
my_dataset = "my_package:my_dataset_loader"
```

## Using Plugins with ExperimentalPlatform

Plugins are auto-discovered. No extra configuration needed:

```python
from feature_forge import ExperimentalPlatform

platform = ExperimentalPlatform()

# Auto-discovers all registered plugins
print(platform.list_baselines())
print(platform.list_models())
print(platform.list_metrics())
print(platform.list_datasets())

# Use discovered plugins in experiments
results = platform.run(
    datasets=["titanic", "my_dataset"],
    baselines=["malmus", "mymethod"],
    models=["xgboost", "catboost"],  # catboost from plugin
)
```

You can also register plugins programmatically:

```python
platform.register_baseline("adhoc", MyMethod)
platform.register_model("custom", create_custom_model)
platform.register_metric("custom_metric", my_metric_fn)
platform.register_dataset("custom_ds", {"source": "local", ...})
```

## Plugin API Versioning

Feature Forge follows **semantic versioning** for its public API. Plugin contracts are stable within major versions.

| Core Version | Plugin API Stability |
|--------------|----------------------|
| `0.1.x` | BaselineProtocol, model/metric/dataset signatures as documented above |
| `0.2.x` | Expected backward compatible; deprecations announced 1 minor version in advance |
| `1.0.0+` | Full backward compatibility for plugin entry point contracts within major version |

**Compatibility Rules:**
- A plugin that works with `feature-forge==0.1.0` will work with any `0.1.x` release.
- Breaking changes to `BaselineProtocol` or entry point signatures will trigger a **minor version bump** (`0.2.0`) during the `0.x` phase, and a **major version bump** (`1.0.0 → 2.0.0`) after `1.0.0`.
- Deprecated methods will emit `DeprecationWarning` for at least one minor version before removal.

**Best Practices for Plugin Authors:**
1. Pin your plugin's `feature-forge` dependency to the compatible minor version: `feature-forge>=0.1.0,<0.2.0`.
2. Import from `feature_forge.baselines.base` only if you need `Baseline` ABC; otherwise, write protocol-compliant classes without any feature_forge imports.
3. Test your plugin against the latest patch release of the target minor version before publishing.
