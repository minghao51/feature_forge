"""Tests for configuration module."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest
from pydantic import SecretStr, ValidationError
from pydantic_settings import SettingsConfigDict

from feature_forge.config import (
    DataflowConfig,
    EvaluationConfig,
    ExecutionEngine,
    HamiltonCacheConfig,
    LLMConfig,
    MemoryConfig,
    RouterConfig,
    Settings,
    TrackerConfig,
)


class TestLLMConfig:
    def test_default_values(self) -> None:
        cfg = LLMConfig()
        assert cfg.model == "deepseek-chat"
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 32768
        assert cfg.agent_max_tokens == 8192
        assert cfg.codegen_max_tokens == 16384
        assert cfg.cache_responses is True

    def test_temperature_validation(self) -> None:
        with pytest.raises(ValidationError, match="temperature"):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValidationError, match="temperature"):
            LLMConfig(temperature=2.1)

    def test_max_tokens_validation(self) -> None:
        with pytest.raises(ValidationError, match="max_tokens"):
            LLMConfig(max_tokens=0)
        with pytest.raises(ValidationError, match="token limits"):
            LLMConfig(agent_max_tokens=0)
        with pytest.raises(ValidationError, match="token limits"):
            LLMConfig(codegen_max_tokens=0)

    def test_api_key_secret(self) -> None:
        cfg = LLMConfig(api_key=SecretStr("sk-secret"))
        # api_key is non-None by construction (SecretStr passed above); the
        # Optional field type hides that from mypy.
        assert cast(SecretStr, cfg.api_key).get_secret_value() == "sk-secret"

    def test_no_environment_side_effects(self) -> None:
        old_values = {
            k: os.environ.get(k)
            for k in ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
        }
        try:
            for key in old_values:
                os.environ.pop(key, None)
            _ = LLMConfig(api_key=SecretStr("sk-secret"))
            assert os.environ.get("DEEPSEEK_API_KEY") is None
            assert os.environ.get("OPENAI_API_KEY") is None
            assert os.environ.get("ANTHROPIC_API_KEY") is None
            assert os.environ.get("GEMINI_API_KEY") is None
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class TestRouterConfig:
    def test_default_strategy(self) -> None:
        cfg = RouterConfig()
        assert cfg.strategy == "hybrid"

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValidationError):
            RouterConfig(strategy="invalid")  # deliberately invalid

    def test_min_agents_validation(self) -> None:
        with pytest.raises(ValidationError, match="min_agents"):
            RouterConfig(min_agents=0)

    def test_warmup_rounds_validation(self) -> None:
        with pytest.raises(ValidationError, match="warmup_rounds"):
            RouterConfig(warmup_rounds=-1)


class TestEvaluationConfig:
    def test_default_values(self) -> None:
        cfg = EvaluationConfig()
        assert cfg.cv_folds == 5
        assert cfg.feature_eval_backend == "threading"

    def test_cv_folds_validation(self) -> None:
        with pytest.raises(ValidationError, match="cv_folds"):
            EvaluationConfig(cv_folds=1)

    def test_feature_eval_backend_validation(self) -> None:
        with pytest.raises(ValidationError, match="feature_eval_backend"):
            EvaluationConfig(feature_eval_backend="invalid")  # deliberately invalid

    def test_max_cv_workers_validation(self) -> None:
        with pytest.raises(ValidationError, match="max_cv_workers"):
            EvaluationConfig(max_cv_workers=0)


class TestSettings:
    def test_default_values(self) -> None:
        # Assert the *code* defaults independent of the local config/settings.yaml
        # (which legitimately overrides llm.model per environment, e.g. a local
        # OpenCode model). Use a yaml-free subclass so the test is portable.
        class _CodeDefaults(Settings):
            # dict() copy of the TypedDict loses SettingsConfigDict; the copy
            # only overrides the allowed key "yaml_file" to None.
            model_config = cast(SettingsConfigDict, dict(Settings.model_config, yaml_file=None))

        settings = _CodeDefaults()
        assert settings.task == "classification"
        assert settings.metric == "auc"
        assert settings.n_rounds == 4
        assert settings.llm.model == "deepseek-chat"
        assert settings.tracker.backend == "none"
        # Hamilton is the sole execution engine (ADR 0016).
        assert settings.dataflow.engine is ExecutionEngine.HAMILTON
        assert settings.dataflow.artifact_policy == "layer_boundaries"
        assert settings.dataflow.cache.enabled is True
        assert settings.dataflow.cache.max_age_days is None
        assert settings.dataflow.cache.max_size_mb is None
        assert settings.dataflow.cache.telemetry_max_events == 10_000

    def test_metric_validation(self) -> None:
        with pytest.raises(ValidationError, match="metric"):
            Settings(metric="invalid")

    def test_n_rounds_validation(self) -> None:
        with pytest.raises(ValidationError, match="n_rounds"):
            Settings(n_rounds=0)

    def test_init_override(self) -> None:
        settings = Settings(task="regression", n_rounds=10)
        assert settings.task == "regression"
        assert settings.n_rounds == 10

    def test_nested_override(self) -> None:
        # pydantic coerces nested dicts into LLMConfig at runtime; this test
        # asserts exactly that behavior, which the synthesized __init__ hides.
        settings = Settings(llm={"model": "gpt-4", "temperature": 0.5})
        assert settings.llm.model == "gpt-4"
        assert settings.llm.temperature == 0.5

    def test_llm_cache_enforced_default(self) -> None:
        settings = Settings()
        assert settings.llm.cache_responses is True

    def test_dataflow_hamilton_derives_layer_policy(self) -> None:
        settings = Settings(dataflow={"engine": "hamilton"})
        assert settings.dataflow.engine is ExecutionEngine.HAMILTON
        assert settings.dataflow.artifact_policy == "layer_boundaries"

    def test_dataflow_environment_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FF_DATAFLOW__CACHE__ENABLED", "false")
        monkeypatch.setenv("FF_DATAFLOW__CACHE__MAX_AGE_DAYS", "14")
        monkeypatch.setenv("FF_DATAFLOW__CACHE__MAX_SIZE_MB", "512")
        monkeypatch.setenv("FF_DATAFLOW__CACHE__TELEMETRY_MAX_EVENTS", "250")
        settings = Settings()
        assert settings.dataflow.engine is ExecutionEngine.HAMILTON
        assert settings.dataflow.cache.enabled is False
        assert settings.dataflow.cache.max_age_days == 14
        assert settings.dataflow.cache.max_size_mb == 512
        assert settings.dataflow.cache.telemetry_max_events == 250

    def test_dataflow_rejects_legacy_engine_dict(self) -> None:
        with pytest.raises(ValidationError, match="FF_DATAFLOW__ENGINE") as excinfo:
            DataflowConfig(engine="legacy")
        assert "legacy" in str(excinfo.value)
        assert "docs/migration_guide.md" in str(excinfo.value)

    def test_dataflow_rejects_legacy_engine_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FF_DATAFLOW__ENGINE", "legacy")
        with pytest.raises(ValidationError, match="FF_DATAFLOW__ENGINE") as excinfo:
            Settings()
        message = str(excinfo.value)
        assert "legacy" in message
        assert "docs/migration_guide.md" in message

    def test_dataflow_rejects_legacy_engine_case_insensitive(self) -> None:
        with pytest.raises(ValidationError, match="legacy"):
            DataflowConfig(engine="LEGACY")

    def test_dataflow_rejects_legacy_artifact_policy(self) -> None:
        with pytest.raises(ValidationError, match="artifact_policy") as excinfo:
            DataflowConfig(artifact_policy="legacy")
        assert "docs/migration_guide.md" in str(excinfo.value)

    def test_dataflow_cache_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            HamiltonCacheConfig(enabled=True, unexpected=True)  # type: ignore[call-arg]

    @pytest.mark.parametrize("field", ["max_age_days", "max_size_mb", "telemetry_max_events"])
    def test_dataflow_cache_rejects_non_positive_limits(self, field: str) -> None:
        with pytest.raises(ValidationError, match=field):
            HamiltonCacheConfig(**{field: 0})


class TestTrackerConfig:
    def test_default_backend(self) -> None:
        cfg = TrackerConfig()
        assert cfg.backend == "none"

    def test_mlflow_backend(self) -> None:
        cfg = TrackerConfig(backend="mlflow")
        assert cfg.backend == "mlflow"


class TestMemoryConfig:
    def test_default_values(self) -> None:
        cfg = MemoryConfig()
        assert cfg.max_size == 100


class TestSelectionLimitConfig:
    def test_max_selected_features_is_validated(self) -> None:
        assert Settings(max_selected_features=3).max_selected_features == 3
        with pytest.raises(ValidationError, match="max_selected_features"):
            Settings(max_selected_features=0)

    def test_legacy_min_effective_input_alias_migrates(self) -> None:
        settings = Settings(min_effective=4)
        assert settings.max_selected_features == 4
        assert "min_effective" not in settings.model_dump()

    def test_prefixed_environment_name_is_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FF_MAX_SELECTED_FEATURES", "7")
        assert Settings(_env_file=None).max_selected_features == 7


class TestEnvFileLoading:
    """Local .env loading (config.py model_config env_file, 2026-09-05)."""

    def test_env_file_supplies_secrets(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FF_LLM__API_KEY=sk-from-file\n")
        # pydantic-settings' _env_file kwarg is invisible to the __init__
        # mypy synthesizes from the model fields (dataclass_transform).
        settings = Settings(_env_file=env_file)
        assert settings.llm.api_key is not None
        assert settings.llm.api_key.get_secret_value() == "sk-from-file"

    def test_real_env_var_beats_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FF_LLM__API_KEY", "sk-from-env")
        env_file = tmp_path / ".env"
        env_file.write_text("FF_LLM__API_KEY=sk-from-file\n")
        settings = Settings(_env_file=env_file)
        # Non-None is guaranteed by the FF_LLM__API_KEY env var set above;
        # the Optional field type cannot express that.
        assert cast(SecretStr, settings.llm.api_key).get_secret_value() == "sk-from-env"

    def test_env_file_beats_yaml_default(self, tmp_path: Path) -> None:
        # yaml default model is hy3; .env override must win over the file source
        env_file = tmp_path / ".env"
        env_file.write_text("FF_LLM__MODEL=deepseek-chat\n")
        settings = Settings(_env_file=env_file)
        assert settings.llm.model == "deepseek-chat"

    def test_no_env_file_by_default_in_tests(self) -> None:
        # Hermetic guard: with the autouse fixture nulling env_file, a bare
        # Settings() must not pick up the developer's local .env secrets.
        settings = Settings()
        assert not (
            settings.llm.api_key and settings.llm.api_key.get_secret_value().startswith("sk-")
        )
