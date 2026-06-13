"""Unit tests for the memory system."""

from __future__ import annotations

import json

import pytest

from feature_forge.llm.base import LLMClient, LLMResponse
from feature_forge.methods.malmas.memory import AgentMemory, ConceptualMemory, MemoryPersistence
from feature_forge.methods.malmas.memory.retrieval import retrieve_top_k


class FakeLLM(LLMClient):
    def __init__(self, response_text: str = "") -> None:
        super().__init__(model="fake", api_key="fake")
        self.response_text = response_text

    @property
    def provider_name(self) -> str:
        return "fake"

    async def _do_complete(self, messages, temperature=0.2, max_tokens=4096, **kwargs):
        return LLMResponse(content=self.response_text, model=self.model)

    async def _do_complete_json(
        self, messages, schema_description, temperature=0.2, max_tokens=4096
    ):
        return json.loads(self.response_text or "{}")


class TestRetrieval:
    def test_retrieve_top_k_ranks_by_overlap(self):
        entries = [
            {"feature_name": "a", "base_columns": ["age", "fare"], "round_idx": 0, "value": 0.05},
            {"feature_name": "b", "base_columns": ["sibsp"], "round_idx": 0, "value": 0.1},
            {"feature_name": "c", "base_columns": ["age"], "round_idx": 1, "value": 0.02},
        ]
        result = retrieve_top_k(entries, current_columns={"age", "fare"}, current_round=2, top_k=3)
        assert result[0]["feature_name"] == "a"
        assert result[1]["feature_name"] == "c"

    def test_retrieve_top_k_respects_top_k(self):
        entries = [
            {"feature_name": f"f{i}", "base_columns": ["age"], "round_idx": 0, "value": 0.05}
            for i in range(10)
        ]
        result = retrieve_top_k(entries, current_columns={"age"}, current_round=0, top_k=3)
        assert len(result) == 3

    def test_retrieve_top_k_empty(self):
        assert retrieve_top_k([], current_columns={"age"}, current_round=0) == []


class TestMemoryPersistence:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "memory.json")
        pers = MemoryPersistence(path)
        data = {"procedural": [{"feature_name": "f1"}]}
        pers.save(data)
        loaded = pers.load()
        assert loaded == data

    def test_load_missing_returns_none(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        pers = MemoryPersistence(path)
        assert pers.load() is None


class TestAgentMemory:
    def test_enforces_max_size_across_tiers(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path, max_size=2)

        for idx in range(4):
            mem.record_procedure(["a"], "t", f"f{idx}", "num", "d", idx)
            mem.record_feedback(f"fb{idx}", "auc", float(idx), True, idx, ["a"], "num")
            mem.record_conceptual(f"rule {idx}")
            mem.record_global_summary(f"summary {idx}")

        mem.save()

        assert [item["feature_name"] for item in mem.procedural] == ["f2", "f3"]
        assert [item["feature_name"] for item in mem.feedback] == ["fb2", "fb3"]
        assert mem.conceptual == ["rule 2", "rule 3"]
        assert mem.global_summary == ["summary 2", "summary 3"]

    def test_allows_reusing_evicted_feature_names(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path, max_size=2)

        for idx in range(3):
            mem.record_procedure(["a"], "t", f"f{idx}", "num", "d", idx)

        mem.record_procedure(["a"], "t", "f0", "num", "d", 99)

        assert [item["feature_name"] for item in mem.procedural] == ["f2", "f0"]

    def test_record_procedure(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path)
        mem.record_procedure(["age"], "square", "age_sq", "numerical", "age squared", 0)
        assert len(mem.procedural) == 1
        assert mem.procedural[0]["feature_name"] == "age_sq"

    def test_record_procedure_dedup(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path)
        mem.record_procedure(["age"], "square", "age_sq", "numerical", "desc", 0)
        mem.record_procedure(["age"], "cube", "age_sq", "numerical", "desc2", 0)
        assert len(mem.procedural) == 1

    def test_record_feedback(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path)
        mem.record_feedback("age_sq", "auc", 0.05, True, 0, ["age"], "numerical")
        assert len(mem.feedback) == 1
        assert mem.feedback[0]["effective"] is True

    def test_summarize_feedback(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path)
        mem.record_feedback("f1", "auc", 0.1, True, 0, ["a"], "num")
        mem.record_feedback("f2", "auc", 0.05, True, 0, ["b"], "num")
        mem.record_feedback("f3", "auc", -0.01, False, 0, ["c"], "num")
        summary = mem.summarize_feedback(top_k=2)
        assert "f1" in summary
        assert "f3" not in summary

    def test_get_positive_negative(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path)
        mem.record_procedure(["a"], "t", "f1", "num", "d", 0)
        mem.record_feedback("f1", "auc", 0.05, True, 0, ["a"], "num")
        mem.record_unused_procedure(["b"], "t", "f2", "num", "d", 0)
        pos, neg = mem.get_positive_negative_features()
        assert pos == ["f1"]
        assert neg == ["f2"]

    def test_compute_stats(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path)
        mem.record_procedure(["age"], "log", "age_log", "numerical", "log age", 0)
        mem.record_feedback("age_log", "auc", 0.05, True, 0, ["age"], "numerical")
        stats = mem.compute_stats(min_effective=1)
        assert "log" in stats["effective_transforms"]

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path)
        mem.record_procedure(["a"], "t", "f1", "num", "d", 0)
        mem.save()

        mem2 = AgentMemory("unary", path)
        assert len(mem2.procedural) == 1

    def test_generate_prompt_section(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path)
        mem.record_feedback("f1", "auc", 0.1, True, 0, ["a"], "num")
        section = mem.generate_prompt_section(use_feedback=True)
        assert "History Feedback" in section
        assert "f1" in section

    def test_retrieve_relevant_returns_top_k(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path)
        for i in range(20):
            mem.record_feedback(f"f{i}", "auc", 0.1, True, 0, ["age"], "num")
        result = mem.retrieve_relevant(current_columns={"age"}, current_round=0, top_k=5)
        lines = result.strip().split("\n")
        assert len(lines) <= 6
        assert "ranked by relevance" in result

    def test_retrieve_relevant_fallback_to_generate(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path)
        result = mem.retrieve_relevant(current_columns={"age"}, current_round=0, top_k=5)
        assert result == mem.generate_prompt_section(use_feedback=True)


class TestConceptualMemory:
    @pytest.mark.asyncio
    async def test_summarize_agent_insufficient_data(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path)
        llm = FakeLLM("rule 1")
        cm = ConceptualMemory(llm)
        result = await cm.summarize_agent(mem, min_effective=1)
        assert "Few valid features" in result

    @pytest.mark.asyncio
    async def test_summarize_agent_with_data(self, tmp_path):
        path = str(tmp_path / "mem.json")
        mem = AgentMemory("unary", path)
        mem.record_procedure(["age"], "log", "age_log", "numerical", "log", 0)
        mem.record_feedback("age_log", "auc", 0.05, True, 0, ["age"], "numerical")
        llm = FakeLLM("Use log transforms for skewed numerical features.")
        cm = ConceptualMemory(llm)
        result = await cm.summarize_agent(mem, min_effective=1)
        assert "log transforms" in result
        assert mem.conceptual_summary == result

    @pytest.mark.asyncio
    async def test_summarize_global(self, tmp_path):
        path1 = str(tmp_path / "mem1.json")
        path2 = str(tmp_path / "mem2.json")
        mem1 = AgentMemory("unary", path1)
        mem2 = AgentMemory("cross", path2)
        llm = FakeLLM("Global rule: combine unary and cross features.")
        cm = ConceptualMemory(llm)
        result = await cm.summarize_global({"unary": mem1, "cross": mem2})
        assert "Global rule" in result
        assert result in mem1.global_summary
        assert result in mem2.global_summary
