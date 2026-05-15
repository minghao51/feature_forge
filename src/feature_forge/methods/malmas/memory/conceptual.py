"""Conceptual memory with LLM summarization."""

from __future__ import annotations

import json
import time
from typing import Any

from feature_forge.llm.base import LLMClient
from feature_forge.methods.malmas.memory.base import AgentMemory
from feature_forge.methods.malmas.memory.prompts import (
    SummarizeAgentParams,
    SummarizeGlobalParams,
)
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

        params = SummarizeAgentParams(
            agent_name=memory.agent_name,
            examples_text=examples_text,
            stats_text=stats_text,
        )
        system_prompt = params.render_system()
        user_prompt = params.render_user()

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
        params = SummarizeGlobalParams(
            combined_prompt=combined_prompt,
            task_description=task_description,
        )
        system_prompt = params.render_system()
        user_prompt = params.render_user()

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
