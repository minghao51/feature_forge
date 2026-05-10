"""Agents: The 6 Specialized Feature Generators — explore, run, and build custom agents."""

import asyncio
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

import feature_forge

warnings.filterwarnings("ignore")

print(f"feature_forge: {feature_forge.__spec__.origin}")


async def main():
    from feature_forge.agents import AgentRegistry

    agents = AgentRegistry.get_builtin_agents()
    print(f"Built-in agents ({len(agents)}):")
    for name, cls in agents.items():
        print(f"  - {name}: {cls.__name__}")

    from feature_forge.agents.router import RouterAgent
    from feature_forge.config import Settings

    router = RouterAgent(Settings())
    caps = router.AGENT_CAPABILITIES

    cap_df = pd.DataFrame.from_dict(caps, orient="index")
    print("\nAgent capabilities:")
    print(cap_df)

    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(X.shape[1])])
    df["target"] = y

    from feature_forge.config import LLMConfig

    settings = Settings(
        task="classification",
        llm=LLMConfig(
            model="deepseek-chat",
            api_key=os.environ.get("FF_LLM__API_KEY", ""),
            max_concurrent_calls=2,
        ),
    )

    from feature_forge.llm.providers.deepseek import DeepSeekProvider

    llm = DeepSeekProvider(
        model=settings.llm.model,
        api_key=settings.llm.api_key.get_secret_value() if settings.llm.api_key else None,
    )

    context = {
        "description": {col: {"type": "numerical"} for col in df.columns if col != "target"},
        "memory": "",
        "round_idx": 0,
        "positive_features": [],
        "negative_features": [],
    }

    agent_results = {}
    for name, cls in agents.items():
        try:
            agent = cls(config=settings, llm_client=llm)
            specs = await agent.generate(
                X=df.drop(columns=["target"]),
                y=df["target"],
                context=context,
            )
            agent_results[name] = {
                "specs": len(specs),
                "class": cls.__name__,
                "error": None,
            }
        except Exception as exc:
            agent_results[name] = {"specs": 0, "class": cls.__name__, "error": str(exc)[:100]}

    results_df = pd.DataFrame.from_dict(agent_results, orient="index")
    print("\nAgent results:")
    print(results_df)

    agent_name = "unary"
    if agent_name in agents and not agent_results[agent_name].get("error"):
        agent = agents[agent_name](config=settings, llm_client=llm)
        specs = await agent.generate(
            X=df.drop(columns=["target"]), y=df["target"], context=context
        )
        print(f"\n{agent_name} generated {len(specs)} specs:")
        for s in specs[:3]:
            print(f"  - {s}")

    from feature_forge.agents import BaseFeatureAgent

    class PolynomialAgent(BaseFeatureAgent):
        prompt_filename = "unary.txt"
        agent_name = "polynomial"

    print(f"\nCustom agent class: {PolynomialAgent.agent_name}")
    print(f"Built-in agents: {list(AgentRegistry.get_builtin_agents().keys())}")

    plot_df = pd.DataFrame.from_dict(agent_results, orient="index")
    if "specs" in plot_df.columns:
        plot_df = plot_df[plot_df["error"].isna()]
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_df["specs"].plot(kind="bar", ax=ax, color="teal")
            ax.set_title("Feature Specifications Generated per Agent")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            plt.savefig("/tmp/02_agents_specs.png", dpi=100)
            plt.close()
            print("Plot saved to /tmp/02_agents_specs.png")
        else:
            print("No successful agent runs to plot.")


if __name__ == "__main__":
    asyncio.run(main())
