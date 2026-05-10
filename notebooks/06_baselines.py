"""Baselines Comparison — run OpenFE, CAAFE, LLM-FE, and Malmus on the same data and compare artifacts."""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

print("Baselines Comparison Demo")


def main():
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(X.shape[1])])
    df["target"] = y

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["target"]), df["target"],
        test_size=0.3, random_state=42
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    from feature_forge.data.registry import DatasetRegistry

    registry = DatasetRegistry()
    print(f"Available datasets: {registry.list()}")

    try:
        ds = registry.load("titanic")
        titanic_train = ds["train"]
        target_col = ds["target"]
        print(f"Titanic train: {titanic_train.shape}, target: {target_col}")
    except Exception as exc:
        print(f"Titanic dataset unavailable (needs Kaggle API): {exc}")

    from feature_forge.baselines import BaselineRegistry

    baselines = BaselineRegistry.get_builtin_baselines()
    print(f"\nAvailable baselines ({len(baselines)}):")
    for name, cls in baselines.items():
        print(f"  - {name}: {cls.__name__}")

    from feature_forge.llm.providers.deepseek import DeepSeekProvider

    llm = DeepSeekProvider(
        model="deepseek-chat",
        api_key=os.environ.get("FF_LLM__API_KEY", ""),
    )

    baseline_instances = {}

    try:
        from feature_forge.baselines.openfe import OpenFEBaseline
        baseline_instances["openfe"] = OpenFEBaseline(n_jobs=1, metric="auc")
        print("OpenFE: instantiated")
    except Exception as exc:
        print(f"OpenFE: skipped — {exc}")

    try:
        from feature_forge.baselines.caafe import CAAFEBaseline
        baseline_instances["caafe"] = CAAFEBaseline(
            llm_client=llm,
            iterations=2,
            variant="unified",
        )
        print("CAAFE: instantiated")
    except Exception as exc:
        print(f"CAAFE: skipped — {exc}")

    try:
        from feature_forge.baselines.llmfe import LLMFEBaseline
        baseline_instances["llmfe"] = LLMFEBaseline(
            llm_client=llm,
            n_features=5,
            mode="single_shot",
        )
        print("LLM-FE: instantiated")
    except Exception as exc:
        print(f"LLM-FE: skipped — {exc}")

    try:
        from feature_forge.baselines.malmus import MalmusBaseline
        baseline_instances["malmus"] = MalmusBaseline(
            llm_client=llm,
            n_features=5,
            mode="single_shot",
        )
        print("Malmus: instantiated")
    except Exception as exc:
        print(f"Malmus: skipped — {exc}")

    print(f"\nInstantiated: {list(baseline_instances.keys())}")

    from feature_forge.artifacts import compare_methods

    results = compare_methods(
        methods=baseline_instances,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
    )

    for name, artifacts in results.items():
        if "error" in artifacts:
            print(f"{name}: ERROR — {artifacts['error'][:120]}")
        else:
            scripts = artifacts.get("generated_scripts", [])
            print(f"{name}: {len(scripts)} scripts generated")

    for name, artifacts in results.items():
        if "error" in artifacts:
            continue
        print(f"\n=== {name} ===")
        scripts = artifacts.get("generated_scripts", [])
        if scripts:
            print(f"First script (truncated):\n{scripts[0][:300]}")
        else:
            meta = artifacts.get("feature_metadata", [])
            print(f"No code scripts, but {len(meta)} metadata entries")

    from feature_forge.artifacts import ArtifactDashboard
    from feature_forge.artifacts.schema import ArtifactBundle

    bundles = {}
    for name, artifacts in results.items():
        if "error" not in artifacts:
            try:
                bundle_kwargs = {
                    "method_name": name,
                    "generated_scripts": artifacts.get("generated_scripts", []) or [],
                    "feature_metadata": artifacts.get("feature_metadata", []) or [],
                    "provenance_records": artifacts.get("provenance_records", []) or [],
                }
                bundles[name] = ArtifactBundle(**bundle_kwargs)
            except Exception as exc:
                print(f"Could not bundle {name}: {exc}")

    if bundles:
        dash = ArtifactDashboard(bundles)
        report_path = "/tmp/feature_forge_dashboard.html"
        dash.save(report_path)
        print(f"\nDashboard saved to {report_path}")
        print(f"Dashboard size: {os.path.getsize(report_path):,} bytes")
    else:
        print("No valid bundles to display")

    counts = {}
    for name, artifacts in results.items():
        if "error" not in artifacts:
            scripts = artifacts.get("generated_scripts", [])
            counts[name] = len(scripts) if scripts else 0

    if counts:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        pd.Series(counts).plot(kind="bar", ax=ax, color="mediumpurple")
        ax.set_title("Features Generated by Baseline Method")
        ax.set_ylabel("Script Count")
        ax.tick_params(axis="x", rotation=30)

        method_names = []
        method_scripts = []
        method_meta = []
        for name, artifacts in results.items():
            if "error" not in artifacts:
                method_names.append(name.upper())
                method_scripts.append(len(artifacts.get("generated_scripts", []) or []))
                method_meta.append(len(artifacts.get("feature_metadata", []) or []))

        ax = axes[1]
        x = np.arange(len(method_names))
        width = 0.35
        ax.bar(x - width / 2, method_scripts, width, label="Scripts", color="mediumpurple")
        ax.bar(x + width / 2, method_meta, width, label="Metadata entries", color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=30)
        ax.set_ylabel("Count")
        ax.set_title("Scripts vs Metadata by Method")
        ax.legend()

        plt.tight_layout()
        plt.savefig("/tmp/06_baselines.png", dpi=120)
        plt.close()
        print("Plot saved to /tmp/06_baselines.png")

    print("\n--- Per-Method Feature Detail ---")
    for name, artifacts in results.items():
        if "error" in artifacts:
            continue
        meta = artifacts.get("feature_metadata", []) or []
        scripts = artifacts.get("generated_scripts", []) or []
        print(f"\n{name.upper()}:")
        print(f"  Scripts: {len(scripts)}")
        if meta:
            meta_df = pd.DataFrame(meta)
            print(f"  Metadata columns: {list(meta_df.columns)}")
            if "gain" in meta_df.columns:
                gains = meta_df["gain"].dropna()
                if not gains.empty:
                    print(f"  Gain stats: mean={gains.mean():.4f}, max={gains.max():.4f}")
            if "name" in meta_df.columns:
                print(f"  Feature names: {meta_df['name'].tolist()[:10]}")
        else:
            print("  No feature metadata")

    if bundles:
        from feature_forge.artifacts.diff import ArtifactDiff

        diff = ArtifactDiff(bundles)
        diff_summary = diff.summary()
        print(f"\n--- Cross-Method Diff ---")
        print(f"Total unique features: {diff_summary['total_unique_features']}")
        print(f"Shared across all: {diff_summary['shared_across_all']}")
        for method, data in diff_summary.get("per_method", {}).items():
            print(f"  {method}: {data['total_features']} features, {data['unique_features']} unique")
    else:
        print("No bundles for diff analysis")


if __name__ == "__main__":
    main()
