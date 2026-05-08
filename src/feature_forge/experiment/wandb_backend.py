"""Weights & Biases experiment tracker backend."""

from __future__ import annotations

import tempfile
from typing import Any

import pandas as pd

from feature_forge.exceptions import TrackingError
from feature_forge.experiment.tracker import ExperimentTracker


class WandBTracker(ExperimentTracker):
    """WandB experiment tracking backend.

    Supports dual code logging:
    - Code artifacts: versioned wandb.Artifact(type='code')
    - Table rows: wandb.Table with code text columns
    """

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        log_code_to_artifact: bool = True,
        log_code_to_table: bool = True,
    ) -> None:
        super().__init__(project, entity)
        self.log_code_to_artifact = log_code_to_artifact
        self.log_code_to_table = log_code_to_table
        self._run: Any = None

    def init_run(self, run_name: str, config: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise TrackingError("wandb not installed. Run: uv pip install wandb") from exc

        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            config=config,
            reinit=True,
        )

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.config.update(params)

    def log_artifact(self, path: str, artifact_type: str = "dataset") -> None:
        if self._run is not None:
            artifact = (
                self._run.use_artifact(path, type=artifact_type)
                if path.startswith("wandb:")
                else None
            )
            if artifact is None:
                art = self._run.Artifact(name=path.split("/")[-1], type=artifact_type)
                art.add_file(path)
                self._run.log_artifact(art)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    def _log_dataframe(self, key: str, df: pd.DataFrame) -> None:
        if self._run is None:
            return
        try:
            import wandb

            table = wandb.Table(dataframe=df)
            self._run.log({key: table})
        except ImportError:
            pass

    def _log_code(self, key: str, code: str) -> None:
        if self._run is None:
            return
        try:
            import os

            import wandb

            if self.log_code_to_artifact:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".py",
                    delete=False,
                    prefix=f"{key}_",
                ) as f:
                    f.write(code)
                    f.flush()
                    art = wandb.Artifact(name=key, type="code")
                    art.add_file(f.name, name=f"{key}.py")
                    self._run.log_artifact(art)
                    os.unlink(f.name)

            if self.log_code_to_table:
                table = wandb.Table(
                    columns=["artifact_key", "code"],
                    data=[[key, code]],
                )
                self._run.log({f"{key}_table": table})
        except ImportError:
            pass
