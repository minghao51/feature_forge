"""Immutable, validated configuration using pydantic-settings.

Configuration priority (highest to lowest):
1. Constructor arguments
2. Environment variables (FF_* prefix, injected by dotenvx)
3. YAML files (config/settings.yaml)

Secrets are managed via dotenvx — ``dotenvx run --`` injects decrypted
values into the environment before Python starts. Non-sensitive defaults
live in ``config/settings.yaml``.

Example:
    >>> from feature_forge.config import Settings
    >>> settings = Settings()
    >>> settings.task
    'classification'
    >>> settings.llm.model
    'deepseek-chat'
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class LLMConfig(BaseModel):
    """LLM provider configuration.

    A single ``api_key`` can be passed to provider clients directly.
    Configuration validation does not mutate global process environment.

    Attributes:
        model: Model identifier (e.g. "deepseek-chat", "gpt-4").
        api_key: Single API key for all LLM providers.
        base_url: Base URL for the API endpoint.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens per response.
        cache_responses: Whether to cache LLM responses.
        max_concurrent_calls: Max concurrent LLM calls.
    """

    model: str = "deepseek-chat"
    provider: Literal["auto", "deepseek", "openai", "anthropic", "litellm"] = "auto"
    api_key: SecretStr | None = None
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    cache_responses: bool = True
    max_concurrent_calls: int = 3

    @field_validator("base_url", "api_key", mode="before")
    @classmethod
    def _empty_string_to_none(cls, v: object) -> object:
        """Coerce empty strings to None for optional fields."""
        return None if v == "" else v

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {v}")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_tokens must be >= 1, got {v}")
        return v


class TrackerConfig(BaseModel):
    """Experiment tracking configuration.

    Attributes:
        backend: Which tracker to use.
        project: Project name in the tracker.
        entity: Team/entity name (WandB only).
    """

    backend: Literal["wandb", "mlflow", "none"] = "wandb"
    project: str = "feature-forge"
    entity: str | None = None


class RouterConfig(BaseModel):
    """Router agent configuration.

    Attributes:
        strategy: How to select agents each round.
        max_agents: Maximum agents to activate (None = dynamic).
        min_agents: Minimum agents to activate.
    """

    strategy: Literal["data_driven", "performance_driven", "hybrid", "llm"] = "hybrid"
    max_agents: int | None = None
    min_agents: int = 1

    @field_validator("min_agents")
    @classmethod
    def _validate_min_agents(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"min_agents must be >= 1, got {v}")
        return v


class MemoryConfig(BaseModel):
    """Memory system configuration.

    Attributes:
        max_size: Maximum entries per memory type.
        persistence_dir: Where to persist memories.
    """

    max_size: int = 100
    persistence_dir: str = "memory_files/agent_memories"


class RetryConfig(BaseModel):
    """LLM call retry configuration.

    Attributes:
        max_retries: Maximum number of retry attempts.
        backoff_base: Base delay in seconds for exponential backoff.
        backoff_max: Maximum delay in seconds between retries.
        backoff_exponent: Exponent for backoff calculation.
    """

    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 30.0
    backoff_exponent: float = 2.0

    @field_validator("max_retries")
    @classmethod
    def _validate_max_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_retries must be >= 0, got {v}")
        return v

    @field_validator("backoff_base")
    @classmethod
    def _validate_backoff_base(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"backoff_base must be > 0, got {v}")
        return v

    @field_validator("backoff_max")
    @classmethod
    def _validate_backoff_max(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"backoff_max must be > 0, got {v}")
        return v


class EvaluationConfig(BaseModel):
    """Feature evaluation configuration.

    Attributes:
        cv_folds: Number of cross-validation folds.
        test_size: Fraction of data for test split.
        fail_on_feature_error: Raise on feature evaluation failure instead of logging.
        fail_on_agent_error: Raise on agent generation failure instead of skipping.
        sandbox_timeout_seconds: Max seconds for sandbox worker execution.
        sandbox_max_memory_mb: Max memory (MB) for sandbox worker process.
        max_candidate_features: Cap on candidate features sent to CV scoring.
    """

    cv_folds: int = 5
    test_size: float = 0.4
    fail_on_feature_error: bool = False
    fail_on_agent_error: bool = False
    sandbox_timeout_seconds: float = 5.0
    sandbox_max_memory_mb: int = 512
    max_candidate_features: int = 50

    @field_validator("cv_folds")
    @classmethod
    def _validate_cv_folds(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"cv_folds must be >= 2, got {v}")
        return v

    @field_validator("test_size")
    @classmethod
    def _validate_test_size(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {v}")
        return v

    @field_validator("sandbox_timeout_seconds")
    @classmethod
    def _validate_sandbox_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"sandbox_timeout_seconds must be > 0, got {v}")
        return v

    @field_validator("sandbox_max_memory_mb")
    @classmethod
    def _validate_sandbox_memory(cls, v: int) -> int:
        if v < 128:
            raise ValueError(f"sandbox_max_memory_mb must be >= 128, got {v}")
        return v

    @field_validator("max_candidate_features")
    @classmethod
    def _validate_max_candidate_features(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_candidate_features must be >= 1, got {v}")
        return v


class Settings(BaseSettings):
    """Root configuration for feature_forge.

    Loads from YAML, environment variables, and constructor args.

    Environment variable examples:
        FF_TASK=regression
        FF_LLM__MODEL=gpt-4
        FF_LLM__API_KEY=sk-...
        FF_TRACKER__PROJECT=my-project
    """

    model_config = SettingsConfigDict(
        env_prefix="FF_",
        env_nested_delimiter="__",
        yaml_file="config/settings.yaml",
        extra="ignore",
    )

    # Core pipeline settings
    task: Literal["classification", "regression"] = "classification"
    metric: str = "auc"
    n_rounds: int = 4
    min_effective: int = 2
    random_state: int = 42
    verbose: int = 1

    # Subsystem configs
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @field_validator("n_rounds")
    @classmethod
    def _validate_n_rounds(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n_rounds must be >= 1, got {v}")
        return v

    @field_validator("metric")
    @classmethod
    def _validate_metric(cls, v: str) -> str:
        valid = {"auc", "acc", "f1", "rmse", "mae", "r2"}
        if v not in valid:
            raise ValueError(f"metric must be one of {valid}, got {v}")
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Define config source priority."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


def get_settings(**overrides: Any) -> Settings:
    """Get settings with optional overrides.

    This is the primary way to obtain configuration in the codebase.
    """
    return Settings(**overrides)
