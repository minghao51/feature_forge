# Experiment Tracking: WandB + MLflow

## Philosophy

Every experiment must be **tracked, versioned, and comparable**. We support both WandB (default, superior visualization) and MLflow (optional, data sovereignty).

## Architecture: Unified Tracker Interface

```python
# src/feature_forge/experiment/tracker.py
from abc import ABC, abstractmethod
from typing import Any

class ExperimentTracker(ABC):
    """Abstract base for experiment tracking backends."""

    @abstractmethod
    def init_run(self, config: dict[str, Any]) -> None:
        """Initialize a new run with configuration."""
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics, optionally with step."""
        pass

    @abstractmethod
    def log_artifact(self, path: str, artifact_type: str) -> None:
        """Log a file or directory as an artifact."""
        pass

    @abstractmethod
    def log_table(self, name: str, dataframe: Any) -> None:
        """Log a table (e.g., feature comparison)."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finalize the run."""
        pass


class TrackerFactory:
    """Create tracker instances based on configuration."""

    _backends = {
        "wandb": "feature_forge.experiment.wandb_backend.WandBTracker",
        "mlflow": "feature_forge.experiment.mlflow_backend.MLflowTracker",
        "none": "feature_forge.experiment.tracker.NoOpTracker",
    }

    @classmethod
    def create(cls, backend: str) -> ExperimentTracker:
        module_path, class_name = cls._backends[backend].rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)()
```

---

## WandB Implementation (Default)

### Why WandB?
- **Free academic tier**: Pro features at no cost
- **W&B Weave**: Best-in-class LLM evaluation and tracing
- **Superior visualization**: Side-by-side run comparison, parallel coordinates
- **Built-in sweeps**: Bayes/Grid/Random hyperparameter search
- **Artifacts**: Versioned datasets and models with lineage

### Implementation

```python
# src/feature_forge/experiment/wandb_backend.py
import wandb
from .tracker import ExperimentTracker

class WandBTracker(ExperimentTracker):
    """Weights & Biases experiment tracker."""

    def __init__(self):
        self.run = None

    def init_run(self, config: dict) -> None:
        self.run = wandb.init(
            project=config.get("tracker", {}).get("project", "feature-forge"),
            entity=config.get("tracker", {}).get("entity"),
            config=config,
            job_type="feature-engineering",
        )

    def log_params(self, params: dict) -> None:
        wandb.config.update(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        wandb.log(metrics, step=step)

    def log_artifact(self, path: str, artifact_type: str) -> None:
        art = wandb.Artifact(
            name=f"{artifact_type}-{wandb.run.id}",
            type=artifact_type,
        )
        if os.path.isfile(path):
            art.add_file(path)
        else:
            art.add_dir(path)
        wandb.log_artifact(art)

    def log_table(self, name: str, dataframe) -> None:
        table = wandb.Table(dataframe=dataframe)
        wandb.log({name: table})

    def finish(self) -> None:
        if self.run:
            self.run.finish()
```

### WandB + Sklearn Integration

```python
# Log sklearn pipeline performance
from sklearn.pipeline import Pipeline
import wandb

with wandb.init(project="feature-forge") as run:
    pipeline = Pipeline([
        ("fe", FeatureForge(task="classification")),
        ("clf", XGBClassifier()),
    ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    wandb.log({"test_accuracy": score})
    wandb.sklearn.plot_classifier(
        pipeline.named_steps["clf"],
        X_train, X_test, y_train, y_test,
        model_name="XGBoost",
    )
```

### WandB Sweeps for Hyperparameter Search

```python
# Define sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "final_auc", "goal": "maximize"},
    "parameters": {
        "n_rounds": {"values": [1, 2, 4, 6]},
        "llm_temperature": {"distribution": "uniform", "min": 0.0, "max": 1.0},
        "router_strategy": {"values": ["data_driven", "performance_driven", "hybrid"]},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 2},
}

sweep_id = wandb.sweep(sweep_config, project="feature-forge")

def train():
    with wandb.init() as run:
        config = run.config
        fe = FeatureForge(
            n_rounds=config.n_rounds,
            router_strategy=config.router_strategy,
        )
        # ... run experiment ...
        wandb.log({"final_auc": auc})

wandb.agent(sweep_id, function=train, count=20)
```

### LLM Cost Tracking with WandB

```python
# Track per-experiment LLM costs
wandb.log({
    "llm_cost_total_usd": 1.25,
    "llm_cost_per_round": [0.30, 0.35, 0.35, 0.25],
    "llm_tokens_total": 45000,
    "llm_tokens_per_agent": {
        "unary": 12000,
        "cross": 15000,
        "aggregation": 8000,
    },
})
```

---

## MLflow Implementation (Optional)

### Why MLflow?
- **Data sovereignty**: Everything stays on your servers
- **Open source**: No vendor lock-in
- **Model registry**: Strong model lifecycle management
- **Local-first**: Works offline

### Implementation

```python
# src/feature_forge/experiment/mlflow_backend.py
import mlflow
from .tracker import ExperimentTracker

class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker."""

    def __init__(self, tracking_uri: str | None = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.run = None

    def init_run(self, config: dict) -> None:
        experiment_name = config.get("tracker", {}).get("project", "feature-forge")
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()
        mlflow.log_params(self._flatten_dict(config))

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, artifact_type: str) -> None:
        mlflow.log_artifact(path)

    def log_table(self, name: str, dataframe) -> None:
        path = f"/tmp/{name}.csv"
        dataframe.to_csv(path, index=False)
        mlflow.log_artifact(path)

    def finish(self) -> None:
        mlflow.end_run()

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten nested dicts for MLflow params."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
```

### MLflow Local Server

```bash
# Start local MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000

# Configure feature_forge to use it
export FF_TRACKER__BACKEND=mlflow
export FF_TRACKER__PROJECT=feature-forge
export MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## No-Op Tracker

For local development or when tracking is disabled:

```python
class NoOpTracker(ExperimentTracker):
    """No-op tracker for when tracking is disabled."""

    def init_run(self, config): pass
    def log_params(self, params): pass
    def log_metrics(self, metrics, step=None): pass
    def log_artifact(self, path, artifact_type): pass
    def log_table(self, name, dataframe): pass
    def finish(self): pass
```

---

## Tracking Schema

### Parameters (Logged Once)

```python
{
    "dataset": "titanic",
    "task": "classification",
    "metric": "auc",
    "n_rounds": 4,
    "llm_model": "deepseek-chat",
    "llm_temperature": 0.2,
    "router_strategy": "hybrid",
    "random_state": 42,
    "agents": ["unary", "cross", "aggregation", "temporal"],
}
```

### Metrics (Logged Per Round)

```python
{
    # Per-round metrics
    "round_1/n_features_generated": 8,
    "round_1/n_effective_features": 3,
    "round_1/avg_feature_gain": 0.025,
    "round_1/best_feature_gain": 0.04,
    "round_1/llm_cost_usd": 0.15,
    "round_1/latency_seconds": 45.2,

    # Cumulative metrics
    "cumulative/n_features_total": 12,
    "cumulative/auc_improvement": 0.05,

    # Final metrics
    "final/base_auc": 0.82,
    "final/malmas_auc": 0.87,
    "final/improvement": 0.05,
}
```

### Artifacts

| Artifact | Type | Content |
|----------|------|---------|
| `generated_features` | dataset | CSV of all generated features |
| `agent_memories` | memory | JSON of all agent memories |
| `feature_importance` | plot | Bar chart of feature gains |
| `router_history` | json | Agent selection per round |

---

## Configuration

```yaml
# config/settings.yaml
tracker:
  backend: "wandb"  # wandb | mlflow | none
  project: "feature-forge"
  entity: "your-wandb-team"  # WandB only
```

```bash
# Environment variables
export WANDB_API_KEY=...
export WANDB_PROJECT=feature-forge
export WANDB_ENTITY=your-team

# Or for MLflow
export MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## Comparison: WandB vs MLflow for feature_forge

| Dimension | WandB (Default) | MLflow (Optional) |
|-----------|----------------|-------------------|
| Setup | Cloud, zero infra | Local server or self-hosted |
| LLM Tracking | W&B Weave (excellent) | AI Gateway + basic tracing |
| Sweeps | Built-in Bayes/Grid | Requires Optuna |
| Visualization | Best-in-class | Basic |
| Artifacts | Versioned with lineage | Manual S3/Azure config |
| Academic | Free Pro tier | Always free |
| Data Sovereignty | Cloud-hosted | Fully local |
| Integration | `wandb.log()` | `mlflow.log_metric()` |

**Recommendation:** Use WandB for daily research (superior UX, free for academics). Use MLflow for sensitive data or production deployments.
