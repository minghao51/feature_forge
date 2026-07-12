"""Experiment tracker factory — creates the right backend from config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from feature_forge.experiment.mlflow_backend import MLflowTracker
from feature_forge.experiment.tracker import ExperimentTracker, NoOpTracker
from feature_forge.experiment.wandb_backend import WandBTracker

if TYPE_CHECKING:
    from feature_forge.config import TrackerConfig


def create_tracker_from_config(config: TrackerConfig) -> ExperimentTracker:
    """Create an experiment tracker from ``TrackerConfig``.

    Maps ``config.backend`` to the corresponding tracker:

    - ``"wandb"`` → :class:`WandBTracker`
    - ``"mlflow"`` → :class:`MLflowTracker`
    - ``"none"`` → :class:`NoOpTracker`

    The SDK (``wandb`` / ``mlflow``) is imported lazily inside each backend's
    methods, so constructing a tracker never requires the package installed.

    Args:
        config: Tracker configuration carrying backend, project, and entity.

    Returns:
        A tracker instance bound to ``config.project`` / ``config.entity``.
    """
    backend = config.backend
    if backend == "wandb":
        return WandBTracker(project=config.project, entity=config.entity)
    if backend == "mlflow":
        return MLflowTracker(project=config.project, entity=config.entity)
    return NoOpTracker(project=config.project)
