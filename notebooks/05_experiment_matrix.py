"""Experiment Matrix & Tracking — Cartesian experiments with ExperimentRunner, Reporter, WandB, and MLflow."""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

warnings.filterwarnings("ignore")
os.environ.setdefault("FF_LOG_LEVEL", "warning")

print("Experiment Matrix & Tracking Demo")


def main():
    from feature_forge.experiment import ExperimentMatrix

    matrix = (
        ExperimentMatrix()
        .datasets(["synthetic_a"])
        .methods({"malmas": ["full"], "openfe": ["openfe"]})
        .seeds([42])
        .models(["xgboost"])
        .rounds([1])
    )

    configs = matrix.generate()
    print(f"Total configurations: {len(configs)}")
    print("First 3 configs:")
    for c in configs[:3]:
        print(f"  {c}")

    from feature_forge.api import MALMASFeatureEngineer
    from feature_forge.config import LLMConfig, Settings
    from feature_forge.llm.providers.deepseek import DeepSeekProvider

    def run_real_experiment(config: dict) -> dict:
        seed = config.get("seed", 42)
        n_rounds = config.get("n_rounds", 1)
        method_variant = config.get("method", "full")

        X, y = make_classification(
            n_samples=200, n_features=5, n_informative=3, random_state=seed
        )
        X_train_df = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(X.shape[1])])
        y_train_s = pd.Series(y)
        X_train_arr, X_test_arr, y_train_arr, y_test_arr = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )
        X_train = pd.DataFrame(X_train_arr, columns=[f"f{i+1}" for i in range(X.shape[1])])
        X_test = pd.DataFrame(X_test_arr, columns=[f"f{i+1}" for i in range(X.shape[1])])
        y_train = pd.Series(y_train_arr)
        y_test = pd.Series(y_test_arr)

        mode_map = {"full": "full", "no_memory": "no_memory", "openfe": "full"}
        mode = mode_map.get(method_variant, "full")

        llm = DeepSeekProvider(
            model="deepseek-chat",
            api_key=os.environ.get("FF_LLM__API_KEY", ""),
        )
        settings = Settings(
            task="classification",
            metric="auc",
            n_rounds=n_rounds,
            random_state=seed,
            llm=LLMConfig(
                model="deepseek-chat",
                api_key=os.environ.get("FF_LLM__API_KEY", ""),
            ),
        )

        fe = MALMASFeatureEngineer(config=settings, llm_client=llm, mode=mode)

        baseline_scores = cross_val_score(
            GradientBoostingClassifier(random_state=seed),
            X_train, y_train, cv=3, scoring="roc_auc"
        )
        baseline_auc = float(baseline_scores.mean())

        fe.fit(pd.DataFrame(X_train), pd.Series(y_train))
        X_train_enhanced = fe.transform(pd.DataFrame(X_train))
        X_test_enhanced = fe.transform(pd.DataFrame(X_test))

        n_new_features = len([c for c in X_train_enhanced.columns if c.startswith("feat_")])

        enhanced_scores = cross_val_score(
            GradientBoostingClassifier(random_state=seed),
            X_train_enhanced, y_train, cv=3, scoring="roc_auc"
        )
        enhanced_auc = float(enhanced_scores.mean())

        return {
            "dataset": config.get("dataset", "unknown"),
            "method": method_variant,
            "seed": seed,
            "n_rounds": n_rounds,
            "model": config.get("model", "xgboost"),
            "baseline_auc": round(baseline_auc, 4),
            "enhanced_auc": round(enhanced_auc, 4),
            "gain": round(enhanced_auc - baseline_auc, 4),
            "n_features": n_new_features,
        }

    def run_simulated_experiment(config: dict):
        seed = config.get("seed", 42)
        n_rounds = config.get("n_rounds", 1)
        method = config.get("method", "unknown")

        np.random.seed(seed)
        baseline = np.random.uniform(0.70, 0.78)
        gain = np.random.uniform(0.01, 0.06) * n_rounds if method == "malmas" else np.random.uniform(0.01, 0.03)
        enhanced = baseline + gain

        return {
            "dataset": config.get("dataset", "unknown"),
            "method": method,
            "seed": seed,
            "n_rounds": n_rounds,
            "model": config.get("model", "xgboost"),
            "baseline_auc": round(baseline, 4),
            "enhanced_auc": round(enhanced, 4),
            "gain": round(gain, 4),
            "n_features": np.random.randint(2, 8) * n_rounds,
        }

    from feature_forge.experiment import ExperimentRunner

    runner = ExperimentRunner(max_workers=1)

    use_real = bool(os.environ.get("FF_LLM__API_KEY"))
    experiment_fn = run_real_experiment if use_real else run_simulated_experiment
    label = "MALMASFeatureEngineer" if use_real else "simulated"

    results = runner.run(configs, experiment_fn)

    results_df = pd.DataFrame(results)
    print(f"Completed {len(results_df)} runs using {label} experiments")
    print(results_df.head())

    from feature_forge.data.registry import DatasetRegistry

    registry = DatasetRegistry()
    print(f"\nAvailable datasets: {registry.list()}")

    try:
        ds = registry.load("titanic")
        titanic_train = ds["train"]
        target_col = ds["target"]
        print(f"Titanic train: {titanic_train.shape}, target: {target_col}")
    except Exception as exc:
        print(f"Titanic dataset unavailable (needs Kaggle API): {exc}")

    from feature_forge.experiment import Reporter

    reporter = Reporter(results)
    md = reporter.to_markdown()
    print(f"\nReporter output (first 800 chars):")
    print(md[:800])

    from feature_forge.experiment import WandBTracker

    try:
        tracker = WandBTracker(
            project="feature-forge-notebooks",
        )
        tracker.init_run(run_name="notebook_demo", config={"demo": True})
        tracker.log_metrics({"baseline_auc": 0.75, "gain": 0.04})
        tracker.finish()
        print("WandB tracking successful")
    except Exception as exc:
        print(f"WandB tracking skipped: {exc}")

    from feature_forge.experiment import MLflowTracker

    try:
        mlflow_tracker = MLflowTracker(project="feature_forge_notebooks")
        mlflow_tracker.init_run(run_name="notebook_demo", config={"demo": True})
        mlflow_tracker.log_metrics({"baseline_auc": 0.75, "gain": 0.04})
        mlflow_tracker.finish()
        print("MLflow tracking successful")
    except Exception as exc:
        print(f"MLflow tracking skipped: {exc}")

    if not results_df.empty and "gain" in results_df.columns:
        pivot = results_df.groupby(["method", "n_rounds"])["gain"].mean().unstack()
        fig, ax = plt.subplots(figsize=(7, 4))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Average Gain by Method and Rounds")
        ax.set_ylabel("AUC Gain")
        ax.tick_params(axis="x", rotation=0)
        plt.legend(title="Rounds")
        plt.tight_layout()
        plt.savefig("/tmp/05_experiment_gains.png", dpi=100)
        plt.close()
        print("Plot saved to /tmp/05_experiment_gains.png")


if __name__ == "__main__":
    main()
