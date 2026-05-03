"""MLflow experiment tracker backend."""

from __future__ import annotations

import tempfile
from typing import Any

import pandas as pd

from feature_forge.exceptions import TrackingError
from feature_forge.experiment.tracker import ExperimentTracker


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracking backend."""

    def __init__(self, project: str, entity: str | None = None, tracking_uri: str | None = None) -> None:
        super().__init__(project, entity)
        self.tracking_uri = tracking_uri
        self._run_id: str | None = None

    def init_run(self, run_name: str, config: dict[str, Any]) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise TrackingError("mlflow not installed. Run: uv pip install mlflow") from exc

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.project)
        run = mlflow.start_run(run_name=run_name)
        self._run_id = run.info.run_id
        for key, value in config.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        try:
            import mlflow
        except ImportError:
            return
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        try:
            import mlflow
        except ImportError:
            return
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_artifact(self, path: str, artifact_type: str = "dataset") -> None:
        try:
            import mlflow
        except ImportError:
            return
        mlflow.log_artifact(path)

    def finish(self) -> None:
        try:
            import mlflow
        except ImportError:
            return
        mlflow.end_run()
        self._run_id = None

    def _log_dataframe(self, key: str, df: pd.DataFrame) -> None:
        try:
            import os

            import mlflow
        except ImportError:
            return
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".parquet", delete=False, prefix=f"{key}_",
        ) as f:
            df.to_parquet(f.name)
            mlflow.log_artifact(f.name, artifact_path=key)
            os.unlink(f.name)

    def _log_code(self, key: str, code: str) -> None:
        try:
            import os

            import mlflow
        except ImportError:
            return
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix=f"{key}_",
        ) as f:
            f.write(code)
            f.flush()
            mlflow.log_artifact(f.name, artifact_path="code")
            os.unlink(f.name)
