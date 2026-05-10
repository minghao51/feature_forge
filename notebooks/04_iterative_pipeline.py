"""Iterative Pipeline & Memory — multi-round feature engineering with procedural, feedback, and conceptual memory."""

import asyncio
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

print("Iterative Pipeline & Memory Demo")


async def main():
    X, y = make_classification(
        n_samples=300, n_features=7, n_informative=4, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"var_{i+1}" for i in range(X.shape[1])])
    df["target"] = y

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["target"]), df["target"],
        test_size=0.3, random_state=42, stratify=df["target"]
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    from feature_forge.data.registry import DatasetRegistry

    registry = DatasetRegistry()
    print(f"Available datasets: {registry.list()}")

    titanic_train = None
    try:
        ds = registry.load("titanic")
        titanic_train = ds["train"]
        target_col = ds["target"]
        print(f"Titanic train: {titanic_train.shape}, target: {target_col}")
    except Exception as exc:
        print(f"Titanic dataset unavailable (needs Kaggle API): {exc}")

    from feature_forge.config import LLMConfig, Settings
    from feature_forge.pipeline.iterative import IterativePipeline
    from feature_forge.llm.providers.deepseek import DeepSeekProvider

    config = Settings(
        task="classification",
        metric="auc",
        n_rounds=2,
        llm=LLMConfig(
            model="deepseek-chat",
            api_key=os.environ.get("FF_LLM__API_KEY", ""),
            max_concurrent_calls=2,
        ),
    )

    llm = DeepSeekProvider(
        model=config.llm.model,
        api_key=config.llm.api_key.get_secret_value() if config.llm.api_key else None,
    )

    pipeline = IterativePipeline(config=config, llm_client=llm)
    result = {}
    try:
        result = await pipeline.run(X_train, y_train, X_test, task_description="binary classification")
        print(f"Pipeline complete: {len(result.get('selected_features', []))} features selected")
    except Exception as exc:
        print(f"Pipeline error: {exc}")

    print(f"\nSelected features: {result.get('selected_features', [])}")
    print(f"Round summaries: {len(result.get('round_summaries', []))}")

    for i, summary in enumerate(result.get("round_summaries", [])):
        print(f"\n--- Round {i} ---")
        print(f"  Agents: {summary.get('agents', [])}")
        print(f"  Features generated: {summary.get('num_features_generated', 0)}")
        print(f"  Features selected: {summary.get('num_features_selected', 0)}")
        print(f"  Baseline: {summary.get('baseline_score', 0):.4f}")

    from feature_forge.memory.base import AgentMemory
    from feature_forge.memory.conceptual import ConceptualMemory

    mem = AgentMemory(agent_name="unary", memory_path="/tmp/test_unary_memory.json")
    mem.record_procedure(
        base_columns=["var_1"],
        transform="log1p",
        feature_name="log_var_1",
        ty="numerical",
        description="log1p transform of var_1",
        round_idx=0,
    )
    mem.record_feedback(
        feature_name="log_var_1",
        metric="auc",
        value=0.03,
        effective=True,
        round_idx=0,
        base=["var_1"],
        ty="numerical",
    )
    mem.save()

    print("\nProcedural memory:")
    print(mem.procedural)
    print("\nFeedback memory:")
    print(mem.feedback)
    print("\nStats:")
    print(mem.stats)

    conceptual = ConceptualMemory(llm)
    try:
        summary = await conceptual.summarize_agent(mem)
        print(f"\nConceptual summary: {summary}")

        global_summary = await conceptual.summarize_global({"unary": mem})
        print(f"Global summary: {global_summary}")
    except Exception as exc:
        print(f"Conceptual memory error: {exc}")

    agent_gains = result.get("agent_gains", {})
    if agent_gains:
        for agent_name, gains_list in agent_gains.items():
            print(f"{agent_name}: {len(gains_list)} gain records")

    rounds = result.get("round_summaries", [])
    if rounds:
        df_rounds = pd.DataFrame([
            {
                "round": i,
                "generated": r.get("num_features_generated", 0),
                "selected": r.get("num_features_selected", 0),
                "baseline": r.get("baseline_score", 0),
            }
            for i, r in enumerate(rounds)
        ])
        fig, ax1 = plt.subplots(figsize=(8, 4))
        x = np.arange(len(df_rounds))
        width = 0.35
        ax1.bar(x - width/2, df_rounds["generated"], width, color="steelblue", label="Generated")
        ax1.bar(x + width/2, df_rounds["selected"], width, color="coral", label="Selected")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Feature Count")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_rounds["round"])
        plt.title("Iterative Pipeline: Features per Round")
        plt.legend()
        plt.tight_layout()
        plt.savefig("/tmp/04_iterative_rounds.png", dpi=100)
        plt.close()
        print("Plot saved to /tmp/04_iterative_rounds.png")


if __name__ == "__main__":
    asyncio.run(main())
