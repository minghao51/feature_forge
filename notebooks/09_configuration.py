"""Configuration & Tracking — pydantic settings, environment overrides, WandB, and MLflow experiment tracking."""

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

print("Configuration & Tracking Demo")


def main():
    from feature_forge.config import Settings

    settings = Settings()
    print(f"Task: {settings.task}")
    print(f"Metric: {settings.metric}")
    print(f"Rounds: {settings.n_rounds}")
    print(f"Model: {settings.llm.model}")
    print(f"Router strategy: {settings.router.strategy}")
    print(f"CV folds: {settings.evaluation.cv_folds}")

    from feature_forge.config import LLMConfig, RouterConfig

    custom = Settings(
        task="regression",
        metric="rmse",
        n_rounds=2,
        llm=LLMConfig(model="gpt-4", temperature=0.5),
        router=RouterConfig(strategy="llm"),
    )
    print(f"\nCustom task: {custom.task}")
    print(f"Custom metric: {custom.metric}")
    print(f"Custom temp: {custom.llm.temperature}")
    print(f"Custom router: {custom.router.strategy}")

    try:
        bad = Settings(llm=LLMConfig(temperature=3.0))
    except Exception as exc:
        print(f"\nTemp validation caught: {exc}")

    try:
        bad = Settings(router=RouterConfig(min_agents=0))
    except Exception as exc:
        print(f"Min agents validation caught: {exc}")

    from pathlib import Path

    yaml_path = Path("config/settings.yaml")
    if yaml_path.exists():
        content = yaml_path.read_text()
        print(f"\nsettings.yaml exists ({len(content)} chars)")
        print("Preview:")
        print("\n".join(content.splitlines()[:15]))
    else:
        print("settings.yaml not found")

    from feature_forge.data import DatasetRegistry

    registry = DatasetRegistry()
    print(f"\nAvailable datasets: {registry.list()}")

    for name in registry.list():
        info = registry.info(name)
        print(f"  {name}: {info.get('task', 'unknown')} — {info.get('source', 'unknown')}")

    from feature_forge.experiment import ExperimentTracker, NoOpTracker

    noop = NoOpTracker(project="test")
    noop.init_run(run_name="test", config={"x": 1})
    noop.log_metrics({"auc": 0.85})
    noop.finish()
    print("\nNoOp tracker: OK (no external calls)")

    from feature_forge.experiment import WandBTracker

    try:
        wandb_tracker = WandBTracker(
            project="feature-forge-notebooks",
        )
        wandb_tracker.init_run(run_name="config_demo", config={
            "task": "classification",
            "model": "deepseek-chat",
        })
        wandb_tracker.log_metrics({"baseline_auc": 0.74, "enhanced_auc": 0.79})
        wandb_tracker.finish()
        print("WandB tracking: OK")
    except Exception as exc:
        print(f"WandB skipped: {exc}")

    from feature_forge.experiment import MLflowTracker

    try:
        mlflow_tracker = MLflowTracker(project="feature_forge_config")
        mlflow_tracker.init_run(run_name="config_demo", config={
            "task": "classification",
            "model": "deepseek-chat",
        })
        mlflow_tracker.log_metrics({"baseline_auc": 0.74, "enhanced_auc": 0.79})
        mlflow_tracker.finish()
        print("MLflow tracking: OK")
    except Exception as exc:
        print(f"MLflow skipped: {exc}")

    hierarchy = {
        "Constructor args": 100,
        "Environment vars": 80,
        ".env file": 60,
        "YAML files": 40,
        "Defaults": 20,
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(hierarchy.keys()), list(hierarchy.values()), color="darkslateblue")
    ax.set_title("Configuration Priority (Highest → Lowest)")
    ax.set_xlabel("Priority")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("/tmp/09_config_hierarchy.png", dpi=100)
    plt.close()
    print("Plot saved to /tmp/09_config_hierarchy.png")


if __name__ == "__main__":
    main()
