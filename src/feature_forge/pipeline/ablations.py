"""Ablation pipeline variants for controlled experiments."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge.agents.base import AgentRegistry
from feature_forge.config import Settings
from feature_forge.llm.base import LLMClient
from feature_forge.pipeline.iterative import IterativePipeline


class NoMemoryPipeline(IterativePipeline):
    """Ablated pipeline without memory context.

    Agents receive no feedback or procedural memory in prompts.
    """

    async def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame | None = None,
        description: dict[str, Any] | None = None,
        task_description: str = "",
    ) -> dict[str, Any]:
        """Run without memory context."""
        self.memories = {}
        result = await super().run(X_train, y_train, X_test, description, task_description)
        return result


class SingleAgentPipeline(IterativePipeline):
    """Ablated pipeline using only a single agent type.

    Bypasses router and always uses the specified agent.
    """

    def __init__(
        self,
        agent_name: str,
        config: Settings,
        llm_client: LLMClient,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, llm_client, **kwargs)
        self.fixed_agent_name = agent_name

    async def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame | None = None,
        description: dict[str, Any] | None = None,
        task_description: str = "",
    ) -> dict[str, Any]:
        """Run with a fixed single agent, bypassing router."""
        # Override router to always return the fixed agent
        agent_classes = AgentRegistry.get_builtin_agents()
        if self.fixed_agent_name not in agent_classes:
            raise ValueError(f"Unknown agent: {self.fixed_agent_name}")

        # Temporarily replace router
        original_select = self.router.select_agents

        async def _fixed_select(*args, **kwargs):
            from feature_forge.types import AgentName

            return [AgentName(self.fixed_agent_name)]

        self.router.select_agents = _fixed_select
        try:
            result = await super().run(X_train, y_train, X_test, description, task_description)
        finally:
            self.router.select_agents = original_select
        return result


class NoRouterPipeline(IterativePipeline):
    """Ablated pipeline without router — uses all agents every round."""

    async def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame | None = None,
        description: dict[str, Any] | None = None,
        task_description: str = "",
    ) -> dict[str, Any]:
        """Run with all agents every round."""

        async def _all_agents(*args, **kwargs):
            from feature_forge.types import AgentName

            return [AgentName(name) for name in self.router.agent_names]

        original_select = self.router.select_agents
        self.router.select_agents = _all_agents
        try:
            result = await super().run(X_train, y_train, X_test, description, task_description)
        finally:
            self.router.select_agents = original_select
        return result
