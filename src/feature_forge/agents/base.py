"""Abstract base class for feature generation agents and registry.

All agents must inherit from Agent and implement `generate()`.
Discovery happens via Python entry points or `AgentRegistry.get_builtin_agents()`.
"""

from __future__ import annotations

import importlib.metadata
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from feature_forge.config import Settings
from feature_forge.exceptions import AgentError
from feature_forge.llm.base import LLMClient
from feature_forge.observability.structlog_config import get_logger
from feature_forge.types import AgentName, FeatureSpec

logger = get_logger(__name__)


class Agent(ABC):
    """Abstract base for feature generation agents.

    Each agent specializes in a specific type of feature transformation
    (unary, cross-compositional, aggregation, temporal, local transform,
    local pattern).

    Attributes:
        name: Unique agent identifier.
        config: Global settings instance.
        llm_client: Async LLM client for generation calls.
    """

    def __init__(
        self,
        name: AgentName,
        config: Settings,
        llm_client: LLMClient,
    ) -> None:
        self.name = name
        self.config = config
        self.llm_client = llm_client

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt template for this agent."""

    @abstractmethod
    async def generate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        context: dict[str, Any],
    ) -> list[FeatureSpec]:
        """Generate feature specifications from input data.

        Args:
            X: Training features.
            y: Training target.
            context: Additional context including:
                - description: Column metadata dict
                - memory: Memory context string
                - round_idx: Current round number
                - positive_features: List of known good features
                - negative_features: List of known bad features

        Returns:
            List of feature specification dicts.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class BaseFeatureAgent(Agent):
    """Base class for feature agents with common LLM interaction pattern.

    Subclasses must define:
    - `prompt_filename`: Name of the prompt file in prompts/
    - `agent_name`: Unique agent identifier string
    """

    prompt_filename: str = ""
    agent_name: str = ""

    def __init__(
        self,
        config: Settings,
        llm_client: LLMClient,
    ) -> None:
        super().__init__(name=AgentName(self.agent_name), config=config, llm_client=llm_client)
        prompt_path = Path(__file__).parent / "../prompts" / self.prompt_filename
        self._system_prompt = (
            prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
        )

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def _build_user_prompt(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        context: dict[str, Any],
    ) -> str:
        """Build the user prompt from context."""
        description = context.get("description", {})
        memory = context.get("memory", "")
        task = context.get("task", self.config.task)
        positive = context.get("positive_features", [])
        negative = context.get("negative_features", [])

        if not description:
            description = self._infer_column_descriptions(X)

        parts: list[str] = []
        parts.append(f"Task: {task}")
        parts.append(f"Dataset columns ({len(X.columns)}):")
        parts.append(json.dumps(description, indent=2, ensure_ascii=False))

        if memory:
            parts.append(f"\nMemory Context:\n{memory}")
        if positive:
            parts.append(f"\nKnown effective features: {positive}")
        if negative:
            parts.append(f"\nKnown ineffective features: {negative}")

        return "\n\n".join(parts)

    @staticmethod
    def _infer_column_descriptions(X: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Generate column descriptions from DataFrame statistics."""
        import numpy as np

        desc: dict[str, dict[str, Any]] = {}
        for col in X.columns:
            col_data = X[col]
            info: dict[str, Any] = {"name": col}
            if pd.api.types.is_numeric_dtype(col_data):
                info["type"] = "numerical"
                info["mean"] = round(float(col_data.mean()), 4)
                info["std"] = round(float(col_data.std()), 4)
                info["min"] = round(float(col_data.min()), 4)
                info["max"] = round(float(col_data.max()), 4)
                info["missing"] = int(col_data.isna().sum())
            else:
                info["type"] = "categorical"
                info["unique"] = int(col_data.nunique())
                info["top"] = str(col_data.mode().iloc[0]) if len(col_data) > 0 else ""
                info["missing"] = int(col_data.isna().sum())
            desc[col] = info
        return desc

    async def generate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        context: dict[str, Any],
    ) -> list[FeatureSpec]:
        """Generate feature specs via LLM."""
        user_prompt = self._build_user_prompt(X, y, context)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        logger.info(
            "agent_generate_start",
            agent=self.name,
            num_columns=len(X.columns),
            round_idx=context.get("round_idx"),
        )
        gen_t0 = time.perf_counter()
        try:
            response = await self.llm_client.complete(
                messages=messages,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
        except Exception as exc:
            logger.error("agent_generate_error", agent=self.name, error=str(exc))
            raise AgentError(f"{self.name} LLM call failed: {exc}") from exc

        specs = self._parse_response(response.content)
        latency_ms = round((time.perf_counter() - gen_t0) * 1000, 1)
        logger.info(
            "agent_generate_complete",
            agent=self.name,
            num_specs=len(specs),
            latency_ms=latency_ms,
        )
        return specs

    def _parse_response(self, content: str) -> list[FeatureSpec]:
        """Parse JSON array of feature specs from LLM response."""
        content = content.strip()
        if content.startswith("```"):
            content = content.removeprefix("```json").removeprefix("```")
            content = content.removesuffix("```").strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("agent_parse_error", agent=self.name, response_preview=content[:200])
            raise AgentError(f"{self.name} invalid JSON: {exc}") from exc

        if not isinstance(data, list):
            raise AgentError(f"{self.name} expected JSON list, got {type(data).__name__}")

        specs: list[FeatureSpec] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            base_cols = item.get("base_columns", [])
            if isinstance(base_cols, str):
                base_cols = [base_cols]
            for feat in item.get("derived_features", []):
                spec: FeatureSpec = {
                    "name": feat.get("name", "unknown"),
                    "type": feat.get("type", "numerical"),
                    "transform": feat.get("transform", ""),
                    "logic": feat.get("logic", ""),
                    "base_columns": base_cols,
                    "agent_name": self.name,
                }
                specs.append(spec)
        return specs


class AgentRegistry:
    """Discover agents via Python entry points.

    Built-in agents are always available. Additional agents can be
    registered by downstream packages via entry points.
    """

    ENTRY_POINT_GROUP = "feature_forge.agents"

    @classmethod
    def discover(cls) -> dict[str, type[Agent]]:
        """Discover all registered agents from entry points."""
        agents: dict[str, type[Agent]] = {}
        for ep in importlib.metadata.entry_points(group=cls.ENTRY_POINT_GROUP):
            agents[ep.name] = ep.load()
        return agents

    @classmethod
    def get_builtin_agents(cls) -> dict[str, type[Agent]]:
        """Return built-in agents without entry point discovery."""
        from feature_forge.agents.aggregation import AggregationConstructAgent
        from feature_forge.agents.cross_compositional import CrossCompositionalAgent
        from feature_forge.agents.local_pattern import LocalPatternAgent
        from feature_forge.agents.local_transform import LocalTransformAgent
        from feature_forge.agents.temporal import TemporalFeatureAgent
        from feature_forge.agents.unary import UnaryFeatureAgent

        return {
            "unary": UnaryFeatureAgent,
            "cross_compositional": CrossCompositionalAgent,
            "aggregation": AggregationConstructAgent,
            "temporal": TemporalFeatureAgent,
            "local_transform": LocalTransformAgent,
            "local_pattern": LocalPatternAgent,
        }

    @classmethod
    def get_all_agents(cls) -> dict[str, type[Agent]]:
        """Return built-in + entry-point registered agents."""
        agents = cls.get_builtin_agents()
        agents.update(cls.discover())
        return agents
