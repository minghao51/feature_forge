"""Evaluation & Sandboxed Execution — cross-validation feature evaluation, sandboxed code execution, and model factory."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

print("Evaluation & Sandbox Demo")


def main():
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(X.shape[1])])
    df["target"] = y

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["target"]), df["target"],
        test_size=0.3, random_state=42, stratify=df["target"]
    )

    from feature_forge.config import Settings
    from feature_forge.evaluation import CVEvaluator

    config = Settings(task="classification", metric="auc")
    evaluator = CVEvaluator(config=config)

    baseline = evaluator.evaluate_baseline(X_train, y_train)
    print(f"Baseline AUC (5-fold CV): {baseline:.4f}")

    feature_df = pd.DataFrame({
        "f1_times_f2": X_train["f1"] * X_train["f2"],
    })

    gain = evaluator.evaluate_feature(
        X_base=X_train,
        y=y_train,
        feature_df=feature_df,
        baseline_score=baseline,
    )
    print(f"Gain from f1*f2: {gain:+.4f}")

    from feature_forge.evaluation import ModelFactory

    factory = ModelFactory()
    for name in ["xgboost", "lightgbm", "random_forest", "catboost"]:
        try:
            model = factory.get_model(name, task="classification")
            print(f"  {name}: {model.__class__.__name__}")
        except Exception as exc:
            print(f"  {name}: unavailable ({exc})")

    from feature_forge.evaluation.metrics import get_metric

    for metric_name in ["auc", "acc", "f1", "rmse", "mae", "r2"]:
        try:
            fn = get_metric(metric_name)
            print(f"  {metric_name}: {fn.__name__}")
        except Exception as exc:
            print(f"  {metric_name}: unavailable ({exc})")

    from feature_forge.evaluation.sandbox import SandboxedExecutor

    sandbox = SandboxedExecutor(
        timeout_seconds=10,
        max_memory_mb=512,
    )

    code_valid = """
import pandas as pd

def generate_features(df):
    return pd.DataFrame({'squared_f1': df['f1'] ** 2})
"""

    output = sandbox.execute(code_valid, X_train.copy())
    print(f"\nSandbox output shape: {output.shape}")
    print(output.head())

    code_bad = """
import os

def generate_features(df):
    os.system('echo pwned')
    return df
"""

    try:
        sandbox.execute(code_bad, X_train.copy())
    except Exception as exc:
        print(f"\nSandbox correctly rejected unsafe code: {type(exc).__name__}")

    features_to_test = {
        "f1_sq": X_train["f1"] ** 2,
        "f1_plus_f2": X_train["f1"] + X_train["f2"],
        "f1_div_f2": X_train["f1"] / (X_train["f2"] + 1e-6),
    }

    eval_results = []
    for feat_name, feat_series in features_to_test.items():
        feat_df = pd.DataFrame({feat_name: feat_series})
        g = evaluator.evaluate_feature(X_train, y_train, feat_df, baseline)
        eval_results.append({"feature": feat_name, "gain": round(g, 4)})

    eval_df = pd.DataFrame(eval_results).sort_values("gain", ascending=False)
    print("\nFeature evaluation results:")
    print(eval_df)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["green" if g > 0 else "red" for g in eval_df["gain"]]
    ax.barh(eval_df["feature"], eval_df["gain"], color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_title("CV Gain per Synthetic Feature")
    ax.set_xlabel("AUC Gain")
    plt.tight_layout()
    plt.savefig("/tmp/07_evaluation_gains.png", dpi=100)
    plt.close()
    print("Plot saved to /tmp/07_evaluation_gains.png")


if __name__ == "__main__":
    main()
