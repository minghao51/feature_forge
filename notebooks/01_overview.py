#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "pandas", "scikit-learn"]
# ///

import marimo

__generated_with = "0.5.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification
    return make_classification, np, pd


@app.cell
def __(make_classification, pd):
    # Generate a synthetic dataset
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=42)
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4", "f5"])
    df["target"] = y
    df.head()
    return df, y


@app.cell
def __(df):
    # Dataset statistics
    df.describe()
    return


@app.cell
def __(df, y):
    from feature_forge.evaluation import CVEvaluator
    from feature_forge.config import Settings

    config = Settings(task="classification", metric="auc")
    evaluator = CVEvaluator(config=config)
    baseline_score = evaluator.evaluate_baseline(df.drop(columns=["target"]), pd.Series(y))
    baseline_score
    return baseline_score, config, evaluator


@app.cell
def __(baseline_score):
    f"Baseline AUC: {baseline_score:.4f}"
    return


@app.cell
def __():
    # Agent Registry
    from feature_forge.agents import AgentRegistry
    agents = AgentRegistry.get_builtin_agents()
    list(agents.keys())
    return AgentRegistry, agents


@app.cell
def __(AgentRegistry):
    # Router Agent capabilities
    from feature_forge.agents.router import RouterAgent
    from feature_forge.config import Settings

    router = RouterAgent(Settings())
    router.AGENT_CAPABILITIES
    return RouterAgent, router


if __name__ == "__main__":
    app.run()
