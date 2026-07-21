"""Documentation helpers for rendering the Silver Hamilton graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from feature_forge.dataflows.driver import (
    GOLD_FINAL_VARS,
    PLATINUM_FINAL_VARS,
    SILVER_FINAL_VARS,
)


def render_silver_dag(driver: Any, output_path: str | Path) -> Path:
    """Render the Silver DAG through Hamilton's optional visualization support."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        driver.visualize_execution(
            SILVER_FINAL_VARS,
            output_file_path=str(destination),
            show_schema=True,
            bypass_validation=True,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional visualization deps
        raise RuntimeError(
            "DAG rendering requires the pipeline extra with Hamilton visualization support"
        ) from exc
    return destination


def render_gold_dag(driver: Any, output_path: str | Path) -> Path:
    """Render the Gold DAG through Hamilton visualization support."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    driver.visualize_execution(
        GOLD_FINAL_VARS,
        output_file_path=str(destination),
        show_schema=True,
        bypass_validation=True,
    )
    return destination


def render_platinum_dag(driver: Any, output_path: str | Path) -> Path:
    """Render the Platinum DAG through Hamilton visualization support."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    driver.visualize_execution(
        PLATINUM_FINAL_VARS,
        output_file_path=str(destination),
        show_schema=True,
        bypass_validation=True,
    )
    return destination


def render_case_dag(driver: Any, output_path: str | Path) -> Path:
    """Render the complete Silver-through-Platinum case dataflow."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    driver.visualize_execution(
        [*SILVER_FINAL_VARS, *GOLD_FINAL_VARS, *PLATINUM_FINAL_VARS],
        output_file_path=str(destination),
        show_schema=True,
        bypass_validation=True,
    )
    return destination
