"""Iterative pipeline with memory, router, and multi-round execution."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge.agents.base import Agent, AgentRegistry
from feature_forge.agents.router import RouterAgent
from feature_forge.config import Settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.llm.base import LLMClient
from feature_forge.memory.base import AgentMemory
from feature_forge.memory.conceptual import ConceptualMemory
from feature_forge.pipeline.core import CodeGenerator, CorePipeline


class IterativePipeline:
    """N-round feature engineering pipeline with router and memory.

    Each round:
    1. Router selects agents
    2. Agent memory provides context
    3. CorePipeline generates and evaluates features
    4. Memory is updated with results
    5. Router performance is updated
    """

    def __init__(
        self,
        config: Settings,
        llm_client: LLMClient,
        router: RouterAgent | None = None,
        evaluator: CVEvaluator | None = None,
        sandbox: SandboxedExecutor | None = None,
        code_generator: CodeGenerator | None = None,
        memory_dir: str = "memory_files/agent_memories",
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.router = router or RouterAgent(config, llm_client)
        self.core = CorePipeline(config, llm_client, evaluator, sandbox, code_generator)
        self.memory_dir = memory_dir
        self.memories: dict[str, AgentMemory] = {}
        self.conceptual_memory = ConceptualMemory(llm_client)
        self.all_feature_codes: list[str] = []
        self.round_artifacts: list[dict[str, Any]] = []

    async def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame | None = None,
        description: dict[str, Any] | None = None,
        task_description: str = "",
    ) -> dict[str, Any]:
        """Run iterative feature engineering for N rounds.

        Returns:
            Dict with keys:
            - X_train_enhanced: Training data with all generated features
            - X_test_enhanced: Test data with all generated features
            - selected_features: List of selected feature names
            - round_summaries: List of per-round summaries
            - agent_gains: Dict of per-agent gain DataFrames
        """
        self.round_artifacts = []
        self.all_feature_codes = []

        X_train_enhanced = X_train.copy()
        X_test_enhanced = X_test.copy() if X_test is not None else None
        round_summaries: list[dict[str, Any]] = []
        all_agent_gains: dict[str, list[pd.DataFrame]] = {}

        for round_idx in range(self.config.n_rounds):
            # 1. Router selects agents
            selected_names = await self.router.select_agents(
                round_idx=round_idx,
                df=X_train_enhanced,
                description=description or {},
                task_description=task_description,
            )

            # 2. Instantiate selected agents
            agent_classes = AgentRegistry.get_builtin_agents()
            agents: list[Agent] = []
            for name in selected_names:
                if name in agent_classes:
                    agents.append(agent_classes[name](self.config, self.llm_client))

            # 3. Build memory context for each agent
            context: dict[str, Any] = {
                "description": description or {},
                "task": self.config.task,
                "round_idx": round_idx,
            }
            agent_contexts = {}
            for agent in agents:
                memory = self._get_memory(agent.name)
                pos, neg = memory.get_positive_negative_features()
                agent_contexts[agent.name] = {
                    **context,
                    "memory": memory.generate_prompt_section(use_feedback=True),
                    "positive_features": pos,
                    "negative_features": neg,
                }

            # 4. Run core pipeline with per-agent contexts
            core_results = await self.core.run(
                agents=agents,
                X_train=X_train_enhanced,
                y_train=y_train,
                X_test=X_test_enhanced,
                context=[agent_contexts.get(a.name, context) for a in agents],
            )

            # 5. Update memories
            for agent in agents:
                memory = self._get_memory(agent.name)
                agent_gain_df = core_results["agent_gains"].get(agent.name, pd.DataFrame())
                for _, row in agent_gain_df.iterrows():
                    fname = row["feature"]
                    gain = row["gain"]
                    spec = next((s for s in core_results["specs"] if s["name"] == fname), None)
                    if spec:
                        memory.record_procedure(
                            base_columns=spec.get("base_columns", []),
                            transform=spec.get("transform", ""),
                            feature_name=fname,
                            ty=spec.get("type", "unknown"),
                            description=spec.get("logic", ""),
                            round_idx=round_idx,
                        )
                        effective = gain > 0
                        memory.record_feedback(
                            feature_name=fname,
                            metric=self.config.metric,
                            value=gain,
                            effective=effective,
                            round_idx=round_idx,
                            base=spec.get("base_columns", []),
                            ty=spec.get("type", "unknown"),
                        )
                        if not effective:
                            memory.record_unused_procedure(
                                base_columns=spec.get("base_columns", []),
                                transform=spec.get("transform", ""),
                                feature_name=fname,
                                ty=spec.get("type", "unknown"),
                                description=spec.get("logic", ""),
                                round_idx=round_idx,
                            )
                memory.save()

                # Update router performance
                if not agent_gain_df.empty:
                    avg_gain = agent_gain_df["gain"].mean()
                    self.router.update_performance(agent.name, avg_gain)

                # Track gains
                if agent.name not in all_agent_gains:
                    all_agent_gains[agent.name] = []
                all_agent_gains[agent.name].append(agent_gain_df)

            # 6. Collect generated code for transform()
            if core_results.get("generated_code"):
                self.all_feature_codes.append(core_results["generated_code"])

            # 7. Append top features to enhanced datasets
            top_train = core_results["top_features_train"]
            if not top_train.empty:
                for col in top_train.columns:
                    if col not in X_train_enhanced.columns:
                        X_train_enhanced[col] = top_train[col]
                if X_test_enhanced is not None:
                    top_test = core_results["top_features_test"]
                    for col in top_test.columns:
                        if col not in X_test_enhanced.columns:
                            X_test_enhanced[col] = top_test[col]

            round_summaries.append(
                {
                    "round": round_idx + 1,
                    "agents": [a.name for a in agents],
                    "baseline_score": core_results["baseline_score"],
                    "num_features_generated": len(core_results["specs"]),
                    "num_features_selected": len(top_train.columns),
                }
            )

            self.round_artifacts.append(
                {
                    "round": round_idx + 1,
                    "generated_code": core_results.get("generated_code", ""),
                    "all_features_train": core_results.get("all_features_train", pd.DataFrame()),
                    "all_features_test": core_results.get("all_features_test", pd.DataFrame()),
                    "selected_features_train": top_train,
                    "selected_features_test": top_test if X_test_enhanced is not None else pd.DataFrame(),
                    "specs": core_results.get("specs", []),
                    "agent_gains": core_results.get("agent_gains", {}),
                    "baseline_score": core_results["baseline_score"],
                    "gains": core_results.get("gains", {}),
                    "agents": [a.name for a in agents],
                }
            )

        # Optionally generate conceptual summaries
        # (skipped by default to avoid extra LLM calls)

        return {
            "X_train_enhanced": X_train_enhanced,
            "X_test_enhanced": X_test_enhanced,
            "selected_features": [
                c for c in X_train_enhanced.columns if c not in X_train.columns
            ],
            "round_summaries": round_summaries,
            "round_artifacts": self.round_artifacts,
            "agent_gains": all_agent_gains,
            "feature_codes": self.all_feature_codes,
        }

    def _get_memory(self, agent_name: str) -> AgentMemory:
        """Get or create agent memory."""
        if agent_name not in self.memories:
            import os
            path = os.path.join(self.memory_dir, f"{agent_name}_memory.json")
            self.memories[agent_name] = AgentMemory(agent_name, path)
        return self.memories[agent_name]
