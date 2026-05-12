"""Conceptual memory with LLM summarization."""

from __future__ import annotations

import json
import time
from typing import Any

from feature_forge.llm.base import LLMClient
from feature_forge.memory.base import AgentMemory
from feature_forge.observability.structlog_config import get_logger

logger = get_logger(__name__)


class ConceptualMemory:
    """Generates LLM-based conceptual summaries from agent memory."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def summarize_agent(
        self,
        memory: AgentMemory,
        min_effective: int = 1,
    ) -> str:
        num_effective = sum(1 for fb in memory.feedback if fb.get("effective", False))
        logger.info(
            "conceptual_summarize_start", agent=memory.agent_name, num_effective=num_effective
        )
        summary_t0 = time.perf_counter()
        if num_effective < min_effective:
            memory.conceptual_summary = (
                "Few valid features, no available information at the moment."
            )
            memory.record_conceptual(memory.conceptual_summary)
            return memory.conceptual_summary

        stats = memory.compute_stats(min_effective=min_effective)
        effective_examples: list[dict[str, Any]] = []
        for fb in memory.feedback:
            if not fb.get("effective", False):
                continue
            proc = next(
                (p for p in memory.procedural if p["feature_name"] == fb["feature_name"]),
                None,
            )
            if proc:
                effective_examples.append(
                    {
                        "feature_name": fb["feature_name"],
                        "transform": proc["transform"],
                        "base_columns": proc["base_columns"],
                        "type": proc.get("type", "unknown"),
                        "gain": fb["value"],
                        "round_idx": fb.get("round_idx", -1),
                        "agent_name": fb.get("agent_name", memory.agent_name),
                    }
                )

        if not effective_examples:
            memory.conceptual_summary = (
                "No effective patterns were found in this agent's recent feature generation."
            )
            memory.record_conceptual(memory.conceptual_summary)
            return memory.conceptual_summary

        examples_text = json.dumps(effective_examples, ensure_ascii=False, indent=2)
        stats_text = json.dumps(stats, ensure_ascii=False, indent=2)

        system_prompt = (
            f"You are {memory.agent_name} agent, an expert feature engineering assistant. "
            "You will receive a list of effective features and statistics about their patterns. "
            "Your task is to generate effective, high-quality conceptual rules using concise language "
            "that can guide future feature generation. Rules should directly reflect the statistics and examples. "
            "Avoid any irrelevant information."
        )
        user_prompt = (
            f"Here are the effective feature examples:\n\n{examples_text}\n\n"
            f"Here are the statistics about effective features:\n\n{stats_text}\n\n"
            "Based on both the examples and the statistics, summarize 1 to 3 concise and actionable "
            "conceptual rules to optimize future feature generation. Rules should be in clear bullet points."
        )

        response = await self.llm_client.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=1024,
        )
        memory.conceptual_summary = response.content
        memory.record_conceptual(memory.conceptual_summary)
        latency_ms = round((time.perf_counter() - summary_t0) * 1000, 1)
        logger.info(
            "conceptual_summarize_complete",
            agent=memory.agent_name,
            summary_length=len(response.content),
            latency_ms=latency_ms,
        )
        return memory.conceptual_summary

    async def summarize_global(
        self,
        memories: dict[str, AgentMemory],
        task_description: str = "",
    ) -> str:
        sections: list[str] = []
        logger.info(
            "conceptual_global_summarize_start",
            num_agents=len(memories),
        )
        global_t0 = time.perf_counter()
        for agent_name, memory in memories.items():
            conceptual = (
                memory.conceptual_summary.strip()
                if memory.conceptual_summary
                else "No conceptual summary available."
            )
            stats = (
                json.dumps(memory.stats, ensure_ascii=False, indent=2)
                if memory.stats
                else "No stats available."
            )
            sections.append(
                f"Agent: {agent_name}:\nStatistics:\n{stats}\nConceptual Summary:\n{conceptual}\n------------------------"
            )

        combined_prompt = "\n\n".join(sections)
        system_prompt = (
            "You are a senior AutoML optimization assistant. "
            "You will receive conceptual summaries and statistics from multiple feature engineering agents. "
            "Your task is to synthesize these into 2 to 5 concise, effective, high-level conceptual rules "
            "that can guide future global feature derivation tasks across all agents. Avoid any irrelevant information."
        )
        user_prompt = (
            f"The description of this dataset is:\n{task_description}\n"
            f"Here are the conceptual summaries and statistics from all agents:\n\n{combined_prompt}\n\n"
            "Based on the above, summarize 2 to 5 concise, actionable, high-level conceptual rules "
            "for optimizing future feature generation across all agents. Rules should be in clear bullet points."
        )

        response = await self.llm_client.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=1024,
        )
        global_summary = response.content
        for memory in memories.values():
            memory.global_summary.append(global_summary)
        latency_ms = round((time.perf_counter() - global_t0) * 1000, 1)
        logger.info(
            "conceptual_global_summarize_complete",
            num_agents=len(memories),
            summary_length=len(global_summary),
            latency_ms=latency_ms,
        )
        return global_summary
