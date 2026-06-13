"""Agent memory system with procedural, feedback, and conceptual memory.

Modeled after MALMAS 3-tier memory architecture:
- Procedural: successful transforms attempted
- Feedback: feature gains/losses with effectiveness flag
- Conceptual: LLM-summarized actionable rules
"""

from __future__ import annotations

from typing import Any

from feature_forge.methods.malmas.memory.persistence import MemoryPersistence
from feature_forge.methods.malmas.memory.retrieval import retrieve_top_k


class AgentMemory:
    """Per-agent memory with procedural, feedback, and conceptual tiers.

    Attributes:
        agent_name: Name of the owning agent.
        memory_path: Path to JSON persistence file.
        procedural: List of successful transform records.
        unused_procedural: List of ineffective transform records.
        feedback: List of evaluation feedback records.
        conceptual: List of LLM-generated rule strings.
        global_summary: List of global conceptual summaries.
        stats: Mechanical statistics computed from feedback.
        conceptual_summary: Latest LLM-generated conceptual summary.
    """

    def __init__(self, agent_name: str, memory_path: str, max_size: int = 100) -> None:
        self.agent_name = agent_name
        self._persistence = MemoryPersistence(memory_path)
        self.max_size = max_size
        self.procedural: list[dict[str, Any]] = []
        self.unused_procedural: list[dict[str, Any]] = []
        self.feedback: list[dict[str, Any]] = []
        self.conceptual: list[str] = []
        self.global_summary: list[str] = []
        self.stats: dict[str, Any] = {}
        self.conceptual_summary: str = ""
        self._procedural_names: set[str] = set()
        self._unused_procedural_names: set[str] = set()
        self._feedback_names: set[str] = set()
        self._load()

    def _load(self) -> None:
        """Load persisted memory state."""
        data = self._persistence.load()
        if data is not None:
            self.procedural = data.get("procedural", [])
            self.unused_procedural = data.get("unused_procedural", [])
            self.feedback = data.get("feedback", [])
            self.conceptual = data.get("conceptual", [])
            self.global_summary = data.get("global_summary", [])
            self.stats = data.get("stats", {})
            self.conceptual_summary = data.get("conceptual_summary", "")
            self._enforce_limits()
            self._rebuild_indices()

    def _trim_sequence(self, values: list[Any]) -> bool:
        if self.max_size <= 0:
            return False
        excess = len(values) - self.max_size
        if excess > 0:
            del values[:excess]
            return True
        return False

    def _enforce_limits(self) -> None:
        """Apply the configured maximum size to all memory tiers."""
        trimmed = False
        trimmed |= self._trim_sequence(self.procedural)
        trimmed |= self._trim_sequence(self.unused_procedural)
        trimmed |= self._trim_sequence(self.feedback)
        self._trim_sequence(self.conceptual)
        self._trim_sequence(self.global_summary)
        if trimmed:
            self._rebuild_indices()

    def save(self) -> None:
        """Persist current memory state."""
        self._enforce_limits()
        self._persistence.save(
            {
                "procedural": self.procedural,
                "unused_procedural": self.unused_procedural,
                "feedback": self.feedback,
                "conceptual": self.conceptual,
                "global_summary": self.global_summary,
                "stats": self.stats,
                "conceptual_summary": self.conceptual_summary,
            }
        )

    def _rebuild_indices(self) -> None:
        """Rebuild lookup indices from loaded data."""
        self._procedural_names = {p["feature_name"] for p in self.procedural}
        self._unused_procedural_names = {p["feature_name"] for p in self.unused_procedural}
        self._feedback_names = {f["feature_name"] for f in self.feedback}

    # ── Procedural Memory ───────────────────────────────────────────

    def record_procedure(
        self,
        base_columns: list[str],
        transform: str,
        feature_name: str,
        ty: str,
        description: str,
        round_idx: int,
    ) -> None:
        """Record a successfully generated feature transform."""
        if feature_name in self._procedural_names:
            return
        self._procedural_names.add(feature_name)
        self.procedural.append(
            {
                "base_columns": base_columns,
                "transform": transform,
                "feature_name": feature_name,
                "type": ty,
                "description": description,
                "agent_name": self.agent_name,
                "round_idx": round_idx,
            }
        )
        if self._trim_sequence(self.procedural):
            self._rebuild_indices()

    def record_unused_procedure(
        self,
        base_columns: list[str],
        transform: str,
        feature_name: str,
        ty: str,
        description: str,
        round_idx: int,
    ) -> None:
        """Record an ineffective feature transform."""
        if feature_name in self._unused_procedural_names:
            return
        self._unused_procedural_names.add(feature_name)
        self.unused_procedural.append(
            {
                "base_columns": base_columns,
                "transform": transform,
                "feature_name": feature_name,
                "type": ty,
                "description": description,
                "agent_name": self.agent_name,
                "round_idx": round_idx,
            }
        )
        if self._trim_sequence(self.unused_procedural):
            self._rebuild_indices()

    # ── Feedback Memory ─────────────────────────────────────────────

    def record_feedback(
        self,
        feature_name: str,
        metric: str,
        value: float,
        effective: bool,
        round_idx: int,
        base: list[str],
        ty: str,
    ) -> None:
        """Record evaluation feedback for a feature."""
        if feature_name in self._feedback_names:
            return
        self._feedback_names.add(feature_name)
        self.feedback.append(
            {
                "feature_name": feature_name,
                "metric": metric,
                "value": value,
                "effective": effective,
                "round_idx": round_idx,
                "agent_name": self.agent_name,
                "base_columns": base,
                "type": ty,
            }
        )
        if self._trim_sequence(self.feedback):
            self._rebuild_indices()

    def summarize_feedback(self, top_k: int = 5) -> str:
        """Summarize top-k effective feedback entries."""
        if not self.feedback:
            return ""
        sorted_fb = sorted(
            [fb for fb in self.feedback if fb.get("effective", False)],
            key=lambda x: x["value"],
            reverse=True,
        )[:top_k]
        return "\n".join(
            [
                f"{fb['feature_name']} → {fb['metric']}: {fb['value']:.4f} (rank {i + 1})"
                for i, fb in enumerate(sorted_fb)
            ]
        )

    def get_positive_negative_features(self) -> tuple[list[str], list[str]]:
        """Return lists of positive (effective) and negative feature names."""
        positive = [fb["feature_name"] for fb in self.feedback if fb.get("effective", False)]
        negative = [item["feature_name"] for item in self.unused_procedural]
        return positive, negative

    # ── Conceptual Memory ───────────────────────────────────────────

    def record_conceptual(self, rule: str) -> None:
        """Add a conceptual rule if not already present."""
        if rule not in self.conceptual:
            self.conceptual.append(rule)
            self._trim_sequence(self.conceptual)

    def record_global_summary(self, summary: str) -> None:
        """Add a global summary while respecting the configured limit."""
        self.global_summary.append(summary)
        self._trim_sequence(self.global_summary)

    def summarize_conceptual(self) -> str:
        """Return all conceptual rules as a string."""
        self._trim_sequence(self.conceptual)
        return "\n".join(self.conceptual)

    # ── Prompt Context Generation ───────────────────────────────────

    def generate_prompt_section(
        self,
        use_procedural: bool = False,
        use_feedback: bool = True,
    ) -> str:
        """Build a memory context string for agent prompts."""
        self._enforce_limits()
        sections: list[str] = []
        if use_feedback and self.feedback:
            sections.append("【History Feedback】\n" + self.summarize_feedback())
        if use_procedural and self.procedural:
            proc_lines = [
                f"Field: {p['base_columns']} → Transform: {p['transform']} → Feature: {p['feature_name']}"
                for p in self.procedural
            ]
            sections.append("【Operations Attempted】\n" + "\n".join(proc_lines))
        return "\n\n".join(sections)

    def retrieve_relevant(
        self,
        current_columns: set[str],
        current_round: int,
        top_k: int = 15,
    ) -> str:
        self._enforce_limits()
        all_feedback = [{**fb, "value": fb.get("value", 0)} for fb in self.feedback]
        ranked = retrieve_top_k(all_feedback, current_columns, current_round, top_k)
        if not ranked:
            return self.generate_prompt_section(use_feedback=True)
        lines = [
            f"{e['feature_name']} → gain: {e.get('value', 0):.4f} (round {e.get('round_idx', 0)}, cols: {e.get('base_columns', [])})"
            for e in ranked
        ]
        return "【History Feedback (ranked by relevance)】\n" + "\n".join(lines)

    # ── Mechanical Statistics ───────────────────────────────────────

    def compute_stats(self, min_effective: int = 1) -> dict[str, Any]:
        """Compute mechanical statistics from feedback for LLM summarization."""
        self._enforce_limits()
        effective_transforms: dict[str, int] = {}
        effective_fields: dict[str, int] = {}
        effective_types: dict[str, int] = {}
        procedural_by_name: dict[str, dict[str, Any]] = {
            p["feature_name"]: p for p in self.procedural
        }

        for fb in self.feedback:
            if not fb.get("effective", False):
                continue
            proc = procedural_by_name.get(fb["feature_name"])
            if proc:
                tf = proc["transform"]
                effective_transforms[tf] = effective_transforms.get(tf, 0) + 1
                base_fields = "-".join(sorted(proc["base_columns"]))
                effective_fields[base_fields] = effective_fields.get(base_fields, 0) + 1
                ty = proc.get("type", "unknown")
                effective_types[ty] = effective_types.get(ty, 0) + 1

        self.stats = {
            "effective_transforms": {
                k: v for k, v in effective_transforms.items() if v >= min_effective
            },
            "effective_fields": {k: v for k, v in effective_fields.items() if v >= min_effective},
            "effective_types": {k: v for k, v in effective_types.items() if v >= min_effective},
        }
        return self.stats
