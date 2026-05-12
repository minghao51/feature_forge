"""Integration tests for dotenvx environment variable injection flow.

These tests simulate what happens when dotenvx injects decrypted secrets
into the environment before Python starts, verifying that Settings correctly
loads from those environment variables.

Tests the configuration priority chain:
1. Constructor arguments (highest)
2. Environment variables (FF_* prefix, injected by dotenvx)
3. YAML files (config/settings.yaml) (lowest)
"""

from __future__ import annotations

import os

import pytest
import yaml

from feature_forge.config import Settings


class TestDotenvxFlow:
    """Test the full config loading flow with dotenvx-injected environment variables."""

    @pytest.fixture
    def clean_env(self):
        """Save and restore environment variables around each test."""
        # Save original env vars that we might modify
        saved_vars = {
            k: os.environ.get(k)
            for k in [
                "FF_TASK",
                "FF_METRIC",
                "FF_N_ROUNDS",
                "FF_LLM__API_KEY",
                "FF_LLM__MODEL",
                "FF_LLM__TEMPERATURE",
                "FF_TRACKER__PROJECT",
                "FF_ROUTER__STRATEGY",
                "DEEPSEEK_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
            ]
        }
        yield
        # Restore original values
        for key, value in saved_vars.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @pytest.fixture
    def settings_yaml_path(self, tmp_path):
        """Create a temporary settings.yaml file for testing."""
        settings_content = {
            "task": "classification",
            "metric": "auc",
            "n_rounds": 4,
            "llm": {
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com",
                "temperature": 0.2,
                "max_tokens": 4096,
                "cache_responses": True,
                "max_concurrent_calls": 3,
            },
            "tracker": {"backend": "wandb", "project": "feature-forge", "entity": None},
            "router": {"strategy": "hybrid", "max_agents": None, "min_agents": 1},
            "memory": {"max_size": 100, "persistence_dir": "memory_files/agent_memories"},
            "evaluation": {
                "cv_folds": 5,
                "test_size": 0.4,
                "fail_on_feature_error": False,
                "fail_on_agent_error": False,
                "sandbox_timeout_seconds": 5.0,
                "sandbox_max_memory_mb": 512,
                "max_candidate_features": 50,
            },
        }
        yaml_path = tmp_path / "settings.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(settings_content, f)
        return yaml_path

    def test_env_var_override_yaml(self, clean_env, settings_yaml_path, monkeypatch):
        """Test that environment variables override YAML defaults."""
        # Simulate dotenvx injection
        monkeypatch.setenv("FF_TASK", "regression")
        monkeypatch.setenv("FF_METRIC", "rmse")
        monkeypatch.setenv("FF_LLM__MODEL", "gpt-4")

        # Patch the YAML file path by modifying the class directly
        original_yaml_file = Settings.model_config["yaml_file"]
        Settings.model_config["yaml_file"] = str(settings_yaml_path)

        try:
            settings = Settings()

            # Env vars should override YAML
            assert settings.task == "regression"
            assert settings.metric == "rmse"
            assert settings.llm.model == "gpt-4"

            # Non-overridden values should come from YAML
            assert settings.n_rounds == 4
            assert settings.llm.temperature == 0.2
        finally:
            # Restore original YAML file path
            Settings.model_config["yaml_file"] = original_yaml_file

    def test_constructor_override_env(self, clean_env, monkeypatch):
        """Test that constructor arguments override environment variables."""
        # Simulate dotenvx injection
        monkeypatch.setenv("FF_TASK", "regression")
        monkeypatch.setenv("FF_METRIC", "rmse")
        monkeypatch.setenv("FF_LLM__API_KEY", "sk-env-key")

        # Constructor args should win
        settings = Settings(task="classification", metric="mae")

        assert settings.task == "classification"
        assert settings.metric == "mae"
        # API key still comes from env since not overridden in constructor
        assert settings.llm.api_key is not None
        assert settings.llm.api_key.get_secret_value() == "sk-env-key"

    def test_ff_llm_api_key_loading(self, clean_env, monkeypatch):
        """Test that FF_LLM__API_KEY is correctly loaded via env var."""
        # Simulate dotenvx injection of API key
        monkeypatch.setenv("FF_LLM__API_KEY", "sk-test-secret-key")

        settings = Settings()

        assert settings.llm.api_key is not None
        secret_value = settings.llm.api_key.get_secret_value()
        assert secret_value == "sk-test-secret-key"

    def test_nested_llm_config_from_env(self, clean_env, monkeypatch):
        """Test that nested LLM config (model, temperature, etc.) loads from env."""
        # Simulate dotenvx injection
        monkeypatch.setenv("FF_LLM__MODEL", "gpt-4")
        monkeypatch.setenv("FF_LLM__TEMPERATURE", "0.7")
        monkeypatch.setenv("FF_LLM__MAX_TOKENS", "8192")
        monkeypatch.setenv("FF_LLM__BASE_URL", "https://api.openai.com/v1")

        settings = Settings()

        assert settings.llm.model == "gpt-4"
        assert settings.llm.temperature == 0.7
        assert settings.llm.max_tokens == 8192
        assert settings.llm.base_url == "https://api.openai.com/v1"

    def test_evaluation_config_from_env(self, clean_env, monkeypatch):
        """Test that evaluation config loads from environment variables."""
        monkeypatch.setenv("FF_EVALUATION__CV_FOLDS", "10")
        monkeypatch.setenv("FF_EVALUATION__TEST_SIZE", "0.2")
        monkeypatch.setenv("FF_EVALUATION__SANDBOX_TIMEOUT_SECONDS", "10.0")

        settings = Settings()

        assert settings.evaluation.cv_folds == 10
        assert settings.evaluation.test_size == 0.2
        assert settings.evaluation.sandbox_timeout_seconds == 10.0

    def test_router_config_from_env(self, clean_env, monkeypatch):
        """Test that router config loads from environment variables."""
        monkeypatch.setenv("FF_ROUTER__STRATEGY", "data_driven")
        monkeypatch.setenv("FF_ROUTER__MIN_AGENTS", "2")

        settings = Settings()

        assert settings.router.strategy == "data_driven"
        assert settings.router.min_agents == 2

    def test_tracker_config_from_env(self, clean_env, monkeypatch):
        """Test that tracker config loads from environment variables."""
        monkeypatch.setenv("FF_TRACKER__BACKEND", "mlflow")
        monkeypatch.setenv("FF_TRACKER__PROJECT", "my-experiment")

        settings = Settings()

        assert settings.tracker.backend == "mlflow"
        assert settings.tracker.project == "my-experiment"

    def test_empty_api_key_becomes_none(self, clean_env, monkeypatch):
        """Test that empty string API key is converted to None."""
        monkeypatch.setenv("FF_LLM__API_KEY", "")

        settings = Settings()

        assert settings.llm.api_key is None

    def test_priority_chain_full(self, clean_env, settings_yaml_path, monkeypatch):
        """Test the full priority chain: constructor > env > YAML."""
        # YAML has: task=classification, metric=auc, llm.model=deepseek-chat
        # Env sets: task=regression, metric=rmse, llm.model=gpt-4
        # Constructor sets: task=regression, metric=mae (overrides env)

        monkeypatch.setenv("FF_TASK", "regression")
        monkeypatch.setenv("FF_METRIC", "rmse")
        monkeypatch.setenv("FF_LLM__MODEL", "gpt-4")

        # Patch the YAML file path by modifying the class directly
        original_yaml_file = Settings.model_config["yaml_file"]
        Settings.model_config["yaml_file"] = str(settings_yaml_path)

        try:
            # Constructor overrides metric but not task
            settings = Settings(metric="mae")

            # task: env overrides YAML
            assert settings.task == "regression"
            # metric: constructor overrides env
            assert settings.metric == "mae"
            # llm.model: env overrides YAML
            assert settings.llm.model == "gpt-4"
            # n_rounds: from YAML (no env or constructor override)
            assert settings.n_rounds == 4
        finally:
            # Restore original YAML file path
            Settings.model_config["yaml_file"] = original_yaml_file

    def test_provider_specific_keys_fallback(self, clean_env, monkeypatch):
        """Test that provider-specific API keys are not auto-propagated.

        This test verifies that when FF_LLM__API_KEY is set, it does NOT
        automatically propagate to provider-specific env vars like
        DEEPSEEK_API_KEY. Providers should receive the key via the
        api_key parameter in their __init__ method, not from the environment.
        """
        # Set FF_LLM__API_KEY
        monkeypatch.setenv("FF_LLM__API_KEY", "sk-unified-key")

        # Ensure provider-specific keys are NOT set
        for key in ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        settings = Settings()

        # The config should have the key
        assert settings.llm.api_key is not None
        secret_value = settings.llm.api_key.get_secret_value()
        assert secret_value == "sk-unified-key"

        # But it should NOT have propagated to provider env vars
        # (this is the behavior we want - no global side effects)
        assert os.environ.get("DEEPSEEK_API_KEY") is None
        assert os.environ.get("OPENAI_API_KEY") is None
        assert os.environ.get("ANTHROPIC_API_KEY") is None
        assert os.environ.get("GEMINI_API_KEY") is None

    def test_secret_str_not_logged(self, clean_env, monkeypatch, caplog):
        """Test that SecretStr values are not accidentally logged."""
        monkeypatch.setenv("FF_LLM__API_KEY", "sk-secret-key")

        settings = Settings()

        # String representation should not reveal the secret
        settings_str = str(settings)
        assert "sk-secret-key" not in settings_str

        # SecretStr get_secret_value() should work
        assert settings.llm.api_key is not None
        secret_value = settings.llm.api_key.get_secret_value()
        assert secret_value == "sk-secret-key"

    def test_get_settings_with_overrides(self, clean_env, monkeypatch):
        """Test the get_settings() helper function with overrides."""
        monkeypatch.setenv("FF_TASK", "regression")
        monkeypatch.setenv("FF_LLM__API_KEY", "sk-env-key")

        from feature_forge.config import get_settings

        # No overrides - uses env
        settings1 = get_settings()
        assert settings1.task == "regression"

        # With overrides - env is overridden
        settings2 = get_settings(task="classification", n_rounds=10)
        assert settings2.task == "classification"
        assert settings2.n_rounds == 10
        # API key still from env
        assert settings2.llm.api_key is not None
        secret_value = settings2.llm.api_key.get_secret_value()
        assert secret_value == "sk-env-key"
