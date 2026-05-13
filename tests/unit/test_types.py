"""Tests for shared type aliases and models."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from feature_forge.types import (
    AgentName,
    DatasetName,
    FeatureSpec,
    MemoryEntry,
    MetricName,
    PromptName,
    RoundNumber,
    Seed,
    T,
    XType,
    YType,
)


class TestFeatureSpecConstruction:
    def test_all_fields(self):
        spec = FeatureSpec(
            name="feat",
            type="categorical",
            transform="df['feat'] = df['a'] + df['b']",
            logic="sum of a and b",
            base_columns=["a", "b"],
            agent_name="agent-1",
        )
        assert spec.name == "feat"
        assert spec.type == "categorical"
        assert spec.transform == "df['feat'] = df['a'] + df['b']"
        assert spec.logic == "sum of a and b"
        assert spec.base_columns == ["a", "b"]
        assert spec.agent_name == "agent-1"

    def test_defaults(self):
        spec = FeatureSpec(name="feat")
        assert spec.type == "numerical"
        assert spec.transform == ""
        assert spec.logic == ""
        assert spec.base_columns == []
        assert spec.agent_name == ""

    def test_name_required(self):
        with pytest.raises(ValidationError, match="name"):
            FeatureSpec()

    def test_base_columns_default_factory(self):
        spec_a = FeatureSpec(name="a")
        spec_b = FeatureSpec(name="b")
        spec_a.base_columns.append("x")
        assert spec_b.base_columns == []


class TestFeatureSpecSerialization:
    def test_model_dump(self):
        spec = FeatureSpec(name="feat", type="categorical", base_columns=["a"])
        dumped = spec.model_dump()
        assert isinstance(dumped, dict)
        assert dumped == {
            "name": "feat",
            "type": "categorical",
            "transform": "",
            "logic": "",
            "base_columns": ["a"],
            "agent_name": "",
        }

    def test_json_round_trip(self):
        spec = FeatureSpec(
            name="feat",
            transform="df['feat'] = 1",
            base_columns=["x", "y"],
        )
        json_str = spec.model_dump_json()
        restored = FeatureSpec.model_validate_json(json_str)
        assert restored == spec

    def test_json_string_round_trip(self):
        spec = FeatureSpec(name="feat")
        json_str = json.dumps(spec.model_dump())
        restored = FeatureSpec(**json.loads(json_str))
        assert restored == spec


class TestFeatureSpecExtraFields:
    def test_extra_fields_allowed(self):
        spec = FeatureSpec(name="feat", extra_key="value")
        assert spec.name == "feat"


class TestNewTypeAliases:
    def test_agent_name(self):
        name = AgentName("agent-1")
        assert isinstance(name, str)
        assert name == "agent-1"

    def test_dataset_name(self):
        name = DatasetName("ds-1")
        assert isinstance(name, str)
        assert name == "ds-1"

    def test_metric_name(self):
        name = MetricName("auc")
        assert isinstance(name, str)
        assert name == "auc"

    def test_prompt_name(self):
        name = PromptName("p-1")
        assert isinstance(name, str)
        assert name == "p-1"

    def test_seed(self):
        s = Seed(42)
        assert isinstance(s, int)
        assert s == 42

    def test_round_number(self):
        r = RoundNumber(3)
        assert isinstance(r, int)
        assert r == 3

    def test_memory_entry(self):
        entry: MemoryEntry = {"key": "value", "num": 1}
        assert isinstance(entry, dict)


class TestTypeVariables:
    def test_t_exists(self):
        assert T.__name__ == "T"

    def test_xtype_bound(self):
        assert XType.__bound__ == pd.DataFrame

    def test_ytype_bound(self):
        assert YType.__bound__ == pd.Series
