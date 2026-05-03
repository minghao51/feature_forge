"""Tests for configuration module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from feature_forge.config import (
    EvaluationConfig,
    LLMConfig,
    MemoryConfig,
    RouterConfig,
    Settings,
    TrackerConfig,
)


class TestLLMConfig:
    def test_default_values(self):
        cfg = LLMConfig()
        assert cfg.model == "deepseek-chat"
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 4096
        assert cfg.cache_responses is True

    def test_temperature_validation(self):
        with pytest.raises(ValidationError, match="temperature"):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValidationError, match="temperature"):
            LLMConfig(temperature=2.1)

    def test_max_tokens_validation(self):
        with pytest.raises(ValidationError, match="max_tokens"):
            LLMConfig(max_tokens=0)

    def test_api_key_secret(self):
        cfg = LLMConfig(api_key="sk-secret")
        assert cfg.api_key.get_secret_value() == "sk-secret"


class TestRouterConfig:
    def test_default_strategy(self):
        cfg = RouterConfig()
        assert cfg.strategy == "hybrid"

    def test_invalid_strategy(self):
        with pytest.raises(ValidationError):
            RouterConfig(strategy="invalid")

    def test_min_agents_validation(self):
        with pytest.raises(ValidationError, match="min_agents"):
            RouterConfig(min_agents=0)


class TestEvaluationConfig:
    def test_default_values(self):
        cfg = EvaluationConfig()
        assert cfg.cv_folds == 5
        assert cfg.test_size == 0.4

    def test_cv_folds_validation(self):
        with pytest.raises(ValidationError, match="cv_folds"):
            EvaluationConfig(cv_folds=1)

    def test_test_size_validation(self):
        with pytest.raises(ValidationError, match="test_size"):
            EvaluationConfig(test_size=0.0)
        with pytest.raises(ValidationError, match="test_size"):
            EvaluationConfig(test_size=1.0)


class TestSettings:
    def test_default_values(self):
        settings = Settings()
        assert settings.task == "classification"
        assert settings.metric == "auc"
        assert settings.n_rounds == 4
        assert settings.llm.model == "deepseek-chat"
        assert settings.tracker.backend == "wandb"

    def test_metric_validation(self):
        with pytest.raises(ValidationError, match="metric"):
            Settings(metric="invalid")

    def test_n_rounds_validation(self):
        with pytest.raises(ValidationError, match="n_rounds"):
            Settings(n_rounds=0)

    def test_init_override(self):
        settings = Settings(task="regression", n_rounds=10)
        assert settings.task == "regression"
        assert settings.n_rounds == 10

    def test_nested_override(self):
        settings = Settings(llm={"model": "gpt-4", "temperature": 0.5})
        assert settings.llm.model == "gpt-4"
        assert settings.llm.temperature == 0.5

    def test_llm_cache_enforced_default(self):
        settings = Settings()
        assert settings.llm.cache_responses is True


class TestTrackerConfig:
    def test_default_backend(self):
        cfg = TrackerConfig()
        assert cfg.backend == "wandb"

    def test_mlflow_backend(self):
        cfg = TrackerConfig(backend="mlflow")
        assert cfg.backend == "mlflow"


class TestMemoryConfig:
    def test_default_values(self):
        cfg = MemoryConfig()
        assert cfg.max_size == 100
