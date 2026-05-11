"""Ablation pipeline variants for controlled experiments."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_forge.agents.base import Agent, AgentRegistry
from feature_forge.config import Settings
from feature_forge.llm.base import LLMClient
from feature_forge.pipeline.iterative import BaseIterativePipeline, IterativePipeline


class NoMemoryPipeline(IterativePipeline):
    """Ablated pipeline without memory context.

    Agents receive no feedback or procedural memory in prompts.
    Inherits from IterativePipeline so router still works,
    but overrides memory to be empty.
    """

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
        for agent in agents:
            agent_gain_df = core_results["agent_gains"].get(agent.name, pd.DataFrame())
            if not agent_gain_df.empty:
                avg_gain = agent_gain_df["gain"].mean()
                self.router.update_performance(agent.name, avg_gain)


class SingleAgentPipeline(BaseIterativePipeline):
    """Ablated pipeline using only a single agent type.

    Inherits from BaseIterativePipeline — no router, no memory.
    Only the specified agent module is imported.
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
        self._agent_cls = AgentRegistry.get_agent(agent_name)

    @property
    def _strategy_label(self) -> str:
        return f"single:{self.fixed_agent_name}"

    async def _select_agents(self, *args, **kwargs) -> list[str]:
        return [self.fixed_agent_name]

    async def _build_agent_context(self, agent, context):
        return context


class NoRouterPipeline(BaseIterativePipeline):
    """Ablated pipeline without router — uses all agents every round.

    Inherits from BaseIterativePipeline — no memory overhead.
    All built-in agents are used every round.
    """

    @property
    def _strategy_label(self) -> str:
        return "no_router"

    async def _select_agents(self, *args, **kwargs) -> list[str]:
        return AgentRegistry.builtin_agent_names()

    async def _build_agent_context(self, agent, context):
        return context
