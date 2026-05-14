"""Router agent for dynamic agent subset selection.

Implements data-driven, performance-driven, hybrid, and LLM-based
selection strategies.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from feature_forge.config import Settings
from feature_forge.llm.base import LLMClient
from feature_forge.observability.structlog_config import get_logger
from feature_forge.prompts import get_registry
from feature_forge.types import AgentName

logger = get_logger(__name__)


class RouterAgent:
    """Selects active agent subset for each iteration.

    Selection strategies:
    - data_driven: Based on dataset characteristics
    - performance_driven: Based on historical gains
    - hybrid: Union of data-driven and performance-driven
    - llm: Uses LLM to make the selection
    """

    AGENT_CAPABILITIES: ClassVar[dict[str, dict[str, Any]]] = {
        "unary": {
            "description": "Generates features from single columns",
            "required_column_types": ["numerical", "categorical"],
            "excluded_if": ["no_single_column_features"],
        },
        "cross_compositional": {
            "description": "Generates cross features between 2+ columns",
            "required_column_types": ["numerical", "categorical"],
            "min_columns": 2,
            "excluded_if": ["single_column_dataset"],
        },
        "aggregation": {
            "description": "Generates aggregation-based features",
            "required_column_types": ["categorical", "groupable"],
            "excluded_if": ["no_categorical_for_grouping"],
        },
        "temporal": {
            "description": "Generates time-based features",
            "required_column_types": ["datetime", "temporal"],
            "excluded_if": ["no_datetime_columns"],
        },
        "local_transform": {
            "description": "Generates local transformation features",
            "required_column_types": ["numerical"],
            "excluded_if": ["no_numerical_columns"],
        },
        "local_pattern": {
            "description": "Generates features based on distributional patterns",
            "required_column_types": ["numerical", "categorical"],
            "requires_enrich": True,
            "excluded_if": [],
        },
    }

    def __init__(
        self,
        config: Settings,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.agent_names = list(self.AGENT_CAPABILITIES.keys())
        self.strategy = config.router.strategy
        self.min_agents = config.router.min_agents or 1
        self.max_agents = config.router.max_agents or len(self.agent_names)
        self.warmup_rounds = 1
        self.use_llm = self.strategy == "llm"

        self.agent_performance: dict[str, list[float]] = {name: [] for name in self.agent_names}
        self.agent_selection_count: dict[str, int] = dict.fromkeys(self.agent_names, 0)
        self.dataset_characteristics: dict[str, Any] | None = None

        self.router_prompt = get_registry().get("router").system

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        description: dict[str, Any],
        enrich_description: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Analyze dataset characteristics for data-driven selection."""
        characteristics: dict[str, Any] = {
            "total_columns": len(df.columns),
            "numerical_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "has_enrich_description": enrich_description is not None,
            "target_column": None,
        }
        for col_name, col_info in description.items():
            if isinstance(col_info, dict):
                col_type = col_info.get("type", "unknown").lower()
                if col_type in ("numerical", "numeric", "float", "int", "integer"):
                    characteristics["numerical_columns"].append(col_name)
                elif col_type in ("categorical", "category", "string", "object"):
                    characteristics["categorical_columns"].append(col_name)
                elif col_type in ("datetime", "date", "time", "temporal"):
                    characteristics["datetime_columns"].append(col_name)
        return characteristics

    def _data_driven_selection(self) -> list[str]:
        """Select agents based on dataset characteristics."""
        if self.dataset_characteristics is None:
            return self.agent_names[: self.max_agents]

        selected: list[str] = []
        chars = self.dataset_characteristics
        for agent_name in self.agent_names:
            capabilities = self.AGENT_CAPABILITIES.get(agent_name, {})
            should_include = True
            excluded_if = capabilities.get("excluded_if", [])
            if "no_datetime_columns" in excluded_if and not chars["datetime_columns"]:
                should_include = False
            if "single_column_dataset" in excluded_if and chars["total_columns"] <= 2:
                should_include = False
            if "no_numerical_columns" in excluded_if and not chars["numerical_columns"]:
                should_include = False
            if (
                "no_categorical_for_grouping" in excluded_if
                and len(chars["categorical_columns"]) < 1
            ):
                should_include = False
            if capabilities.get("requires_enrich") and not chars["has_enrich_description"]:
                should_include = False
            if should_include:
                selected.append(agent_name)

        if len(selected) < self.min_agents:
            for agent in self.agent_names:
                if agent not in selected and len(selected) < self.min_agents:
                    selected.append(agent)
        return selected[: self.max_agents]

    def _performance_driven_selection(self) -> list[str]:
        """Select agents based on historical performance."""
        if not any(self.agent_performance.values()):
            return self.agent_names[: self.max_agents]

        avg_performance = {
            name: (sum(gains) / len(gains) if gains else float("-inf"))
            for name, gains in self.agent_performance.items()
        }
        sorted_agents = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)
        selected: list[str] = []
        for agent_name, avg_gain in sorted_agents:
            if avg_gain >= 0.0:
                selected.append(agent_name)
            elif len(selected) < self.min_agents:
                selected.append(agent_name)
            if len(selected) >= self.max_agents:
                break
        return selected

    def _hybrid_selection(self) -> list[str]:
        """Union of data-driven and performance-driven selections."""
        data_selected = set(self._data_driven_selection())
        perf_selected = set(self._performance_driven_selection())
        in_both = list(data_selected & perf_selected)
        only_data = list(data_selected - perf_selected)
        only_perf = list(perf_selected - data_selected)
        selected = in_both + only_data + only_perf
        if len(selected) > self.max_agents:
            selected = selected[: self.max_agents]
        if len(selected) < self.min_agents:
            remaining = [a for a in self.agent_names if a not in selected]
            selected.extend(remaining[: self.min_agents - len(selected)])
        return selected

    async def _llm_based_selection(
        self,
        round_idx: int,
        description: dict[str, Any] | None = None,
        task_description: str | None = None,
    ) -> list[str]:
        """Use LLM to select agents."""
        if self.llm_client is None or not self.router_prompt:
            return self._hybrid_selection()

        context = self._build_selection_context(round_idx, description, task_description)
        messages = [
            {"role": "system", "content": self.router_prompt},
            {"role": "user", "content": context},
        ]
        try:
            response = await self.llm_client.complete_json(
                messages=messages,
                schema_description='{"agents":["unary","cross_compositional"]}',
                temperature=0.3,
                max_tokens=256,
            )
            if isinstance(response, dict):
                selected = response.get("agents", [])
            elif isinstance(response, list):
                selected = response
            else:
                selected = []
            if isinstance(selected, list):
                valid = [a for a in selected if isinstance(a, str) and a in self.agent_names]
                if valid:
                    return valid
                logger.warning("router_llm_selection_empty_valid", selected=selected)
            else:
                logger.warning(
                    "router_llm_selection_invalid_type", response_type=type(response).__name__
                )
        except Exception as exc:
            logger.warning("router_llm_selection_failed", error=str(exc)[:200])
        return self._hybrid_selection()

    def _build_selection_context(
        self,
        round_idx: int,
        description: dict[str, Any] | None,
        task_description: str | None,
    ) -> str:
        """Build prompt context for LLM-based selection."""
        parts: list[str] = [f"Current iteration: Round {round_idx + 1}"]
        if self.dataset_characteristics:
            chars = self.dataset_characteristics
            parts.append("\nDataset Characteristics:")
            parts.append(f"- Total columns: {chars.get('total_columns', 0)}")
            parts.append(f"- Numerical: {len(chars.get('numerical_columns', []))}")
            parts.append(f"- Categorical: {len(chars.get('categorical_columns', []))}")
            parts.append(f"- Datetime: {len(chars.get('datetime_columns', []))}")
        if any(self.agent_performance.values()):
            parts.append("\nAgent Performance History:")
            for name in self.agent_names:
                gains = self.agent_performance.get(name, [])
                if gains:
                    avg = sum(gains) / len(gains)
                    parts.append(f"- {name}: {avg:.4f} ({len(gains)} rounds)")
                else:
                    parts.append(f"- {name}: No data yet")
        parts.append("\nAvailable Agents:")
        for name, caps in self.AGENT_CAPABILITIES.items():
            parts.append(f"- {name}: {caps['description']}")
        if task_description:
            parts.append(f"\nTask Description: {task_description}")
        return "\n".join(parts)

    async def select_agents(
        self,
        round_idx: int,
        df: pd.DataFrame | None = None,
        description: dict[str, Any] | None = None,
        enrich_description: dict[str, Any] | None = None,
        task_description: str | None = None,
    ) -> list[AgentName]:
        if round_idx < self.warmup_rounds:
            selected = self.agent_names
        else:
            if self.dataset_characteristics is None and df is not None:
                self.dataset_characteristics = self.analyze_dataset(
                    df, description or {}, enrich_description
                )
            if self.use_llm and self.llm_client is not None:
                selected = await self._llm_based_selection(round_idx, description, task_description)
            elif self.strategy == "data_driven":
                selected = self._data_driven_selection()
            elif self.strategy == "performance_driven":
                selected = self._performance_driven_selection()
            else:
                selected = self._hybrid_selection()

        for name in selected:
            self.agent_selection_count[name] += 1

        logger.info(
            "router_select_agents",
            strategy=self.strategy,
            round_idx=round_idx,
            selected_agents=selected,
        )
        return [AgentName(name) for name in selected]

    def update_performance(self, agent_name: str, gain: float) -> None:
        if agent_name in self.agent_performance:
            self.agent_performance[agent_name].append(gain)
            self.agent_performance[agent_name] = self.agent_performance[agent_name][-10:]
        avg_gain = (
            sum(self.agent_performance.get(agent_name, []))
            / len(self.agent_performance.get(agent_name, []))
            if self.agent_performance.get(agent_name)
            else 0.0
        )
        logger.debug(
            "router_performance_update",
            agent=agent_name,
            gain=round(gain, 6),
            avg_gain=round(avg_gain, 6),
        )

    def get_summary(self) -> dict[str, Any]:
        """Return router summary statistics."""
        return {
            "selection_counts": self.agent_selection_count.copy(),
            "average_performance": {
                name: (sum(gains) / len(gains) if gains else 0.0)
                for name, gains in self.agent_performance.items()
            },
            "strategy": self.strategy,
            "dataset_characteristics": self.dataset_characteristics,
        }
