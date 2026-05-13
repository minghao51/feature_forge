"""Iterative pipeline with memory, router, and multi-round execution."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pandas as pd

from feature_forge.agents.base import Agent, AgentRegistry
from feature_forge.agents.router import RouterAgent
from feature_forge.config import Settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.llm.base import LLMClient
from feature_forge.memory.base import AgentMemory
from feature_forge.observability.structlog_config import get_logger
from feature_forge.pipeline.core import CodeGenerator, CorePipeline

if TYPE_CHECKING:
    from feature_forge.memory.conceptual import ConceptualMemory

logger = get_logger(__name__)


class BaseIterativePipeline:
    """Lightweight N-round pipeline without router or memory.

    Subclasses add routing and memory. This base only handles
    round iteration, core pipeline execution, and result collection.

    Each round:
    1. Subclass selects agents via _select_agents()
    2. Subclass builds per-agent context via _build_agent_context()
    3. CorePipeline generates and evaluates features
    4. Subclass handles post-round via _post_round()
    5. Results are collected
    """

    def __init__(
        self,
        config: Settings,
        llm_client: LLMClient,
        evaluator: CVEvaluator | None = None,
        sandbox: SandboxedExecutor | None = None,
        code_generator: CodeGenerator | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.core = CorePipeline(config, llm_client, evaluator, sandbox, code_generator)
        self.all_feature_codes: list[str] = []
        self.round_artifacts: list[dict[str, Any]] = []

    async def _select_agents(
        self,
        round_idx: int,
        X_train: pd.DataFrame,
        description: dict[str, Any],
        task_description: str,
    ) -> list[str]:
        raise NotImplementedError

    async def _build_agent_context(
        self,
        agent: Agent,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return context

    async def _post_round(
        self,
        agents: list[Agent],
        core_results: dict[str, Any],
        round_idx: int,
    ) -> None:
        pass

    @property
    def _strategy_label(self) -> str:
        return "base"

    async def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame | None = None,
        description: dict[str, Any] | None = None,
        task_description: str = "",
    ) -> dict[str, Any]:
        self.round_artifacts = []
        self.all_feature_codes = []

        iterative_t0 = time.perf_counter()
        logger.info(
            "iterative_pipeline_start",
            n_rounds=self.config.n_rounds,
            task=self.config.task,
            strategy=self._strategy_label,
        )

        X_train_enhanced = X_train.copy()
        X_test_enhanced = X_test.copy() if X_test is not None else None
        round_summaries: list[dict[str, Any]] = []
        all_agent_gains: dict[str, list[pd.DataFrame]] = {}

        for round_idx in range(self.config.n_rounds):
            round_t0 = time.perf_counter()
            logger.info("round_start", round_idx=round_idx, total_rounds=self.config.n_rounds)

            selected_names = await self._select_agents(
                round_idx,
                X_train_enhanced,
                description or {},
                task_description,
            )

            agents: list[Agent] = []
            for name in selected_names:
                try:
                    agent_cls = AgentRegistry.get_agent(name)
                    agents.append(agent_cls(self.config, self.llm_client))  # type: ignore[call-arg,arg-type]
                except ValueError:
                    logger.warning("unknown_agent_skipped", agent=name)

            logger.info(
                "agents_selected",
                round_idx=round_idx,
                agents=[a.name for a in agents],
                strategy=self._strategy_label,
            )

            base_context: dict[str, Any] = {
                "description": description or {},
                "task": self.config.task,
                "round_idx": round_idx,
            }

            agent_contexts = []
            for agent in agents:
                ctx = await self._build_agent_context(agent, base_context)
                agent_contexts.append(ctx)

            core_results = await self.core.run(
                agents=agents,
                X_train=X_train_enhanced,
                y_train=y_train,
                X_test=X_test_enhanced,
                context=agent_contexts,
            )

            await self._post_round(agents, core_results, round_idx)

            for agent in agents:
                agent_gain_df = core_results["agent_gains"].get(agent.name, pd.DataFrame())
                if agent.name not in all_agent_gains:
                    all_agent_gains[agent.name] = []
                all_agent_gains[agent.name].append(agent_gain_df)

            if core_results.get("generated_code"):
                self.all_feature_codes.append(core_results["generated_code"])

            top_train = core_results["top_features_train"]
            top_test = core_results.get("top_features_test", pd.DataFrame())
            if not top_train.empty:
                for col in top_train.columns:
                    if col not in X_train_enhanced.columns:
                        X_train_enhanced[col] = top_train[col]
                if X_test_enhanced is not None:
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
                    "selected_features_test": top_test
                    if X_test_enhanced is not None
                    else pd.DataFrame(),
                    "specs": core_results.get("specs", []),
                    "agent_gains": core_results.get("agent_gains", {}),
                    "baseline_score": core_results["baseline_score"],
                    "gains": core_results.get("gains", {}),
                    "agents": [a.name for a in agents],
                }
            )

            round_latency_ms = round((time.perf_counter() - round_t0) * 1000, 1)
            logger.info(
                "round_complete",
                round_idx=round_idx,
                features_generated=len(core_results["specs"]),
                features_selected=len(top_train.columns),
                baseline_score=core_results["baseline_score"],
                latency_ms=round_latency_ms,
            )

        total_latency_ms = round((time.perf_counter() - iterative_t0) * 1000, 1)
        selected_features = [c for c in X_train_enhanced.columns if c not in X_train.columns]
        logger.info(
            "iterative_pipeline_complete",
            total_rounds=self.config.n_rounds,
            total_features=len(selected_features),
            latency_ms=total_latency_ms,
        )

        return {
            "X_train_enhanced": X_train_enhanced,
            "X_test_enhanced": X_test_enhanced,
            "selected_features": selected_features,
            "round_summaries": round_summaries,
            "round_artifacts": self.round_artifacts,
            "agent_gains": all_agent_gains,
            "feature_codes": self.all_feature_codes,
        }


class IterativePipeline(BaseIterativePipeline):
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
        super().__init__(config, llm_client, evaluator, sandbox, code_generator)
        self.router = router or RouterAgent(config, llm_client)
        self.memory_dir = memory_dir
        self.memories: dict[str, AgentMemory] = {}
        self._conceptual_memory: ConceptualMemory | None = None

    @property
    def _strategy_label(self) -> str:
        return self.router.strategy

    @property
    def conceptual_memory(self) -> ConceptualMemory:
        if self._conceptual_memory is None:
            from feature_forge.memory.conceptual import ConceptualMemory

            self._conceptual_memory = ConceptualMemory(self.llm_client)
        return self._conceptual_memory

    async def _select_agents(
        self,
        round_idx: int,
        X_train: pd.DataFrame,
        description: dict[str, Any],
        task_description: str,
    ) -> list[str]:
        selected = await self.router.select_agents(
            round_idx=round_idx,
            df=X_train,
            description=description,
            task_description=task_description,
        )
        return [str(n) for n in selected]

    async def _build_agent_context(
        self,
        agent: Agent,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        memory = self._get_memory(agent.name)
        pos, neg = memory.get_positive_negative_features()
        return {
            **context,
            "memory": memory.generate_prompt_section(use_feedback=True),
            "positive_features": pos,
            "negative_features": neg,
        }

    async def _post_round(
        self,
        agents: list[Agent],
        core_results: dict[str, Any],
        round_idx: int,
    ) -> None:
        for agent in agents:
            memory = self._get_memory(agent.name)
            agent_gain_df = core_results["agent_gains"].get(agent.name, pd.DataFrame())
            for _, row in agent_gain_df.iterrows():
                fname = row["feature"]
                gain = row["gain"]
                spec = next((s for s in core_results["specs"] if s.name == fname), None)
                if spec:
                    memory.record_procedure(
                        base_columns=spec.base_columns,
                        transform=spec.transform,
                        feature_name=fname,
                        ty=spec.type,
                        description=spec.logic,
                        round_idx=round_idx,
                    )
                    effective = gain > 0
                    memory.record_feedback(
                        feature_name=fname,
                        metric=self.config.metric,
                        value=gain,
                        effective=effective,
                        round_idx=round_idx,
                        base=spec.base_columns,
                        ty=spec.type,
                    )
                    if not effective:
                        memory.record_unused_procedure(
                            base_columns=spec.base_columns,
                            transform=spec.transform,
                            feature_name=fname,
                            ty=spec.type,
                            description=spec.logic,
                            round_idx=round_idx,
                        )
            memory.save()

            if not agent_gain_df.empty:
                avg_gain = agent_gain_df["gain"].mean()
                self.router.update_performance(agent.name, avg_gain)

    def _get_memory(self, agent_name: str) -> AgentMemory:
        if agent_name not in self.memories:
            import os

            path = os.path.join(self.memory_dir, f"{agent_name}_memory.json")
            self.memories[agent_name] = AgentMemory(agent_name, path)
            logger.debug("agent_memory_initialized", agent=agent_name, path=path)
        return self.memories[agent_name]
