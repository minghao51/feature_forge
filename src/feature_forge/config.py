"""Immutable, validated configuration using pydantic-settings.

Configuration priority (highest to lowest):
1. Constructor arguments
2. Environment variables (FF_* prefix)
3. Encrypted .env file (managed by dotenvx, decrypted at runtime)
4. YAML files (config/settings.yaml)

Runtime loading:
    Use `dotenvx run --` to decrypt .env before Python starts:
        dotenvx run -- uv run python script.py
        dotenvx run -- uv run pytest tests/

    pydantic-settings reads the decrypted .env automatically.
    No `dotenv.load_dotenv()` needed — dotenvx handles injection.

Example:
    >>> from feature_forge.config import Settings
    >>> settings = Settings()
    >>> settings.task
    'classification'
    >>> settings.llm.model
    'deepseek-chat'
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class LLMConfig(BaseModel):
    """LLM provider configuration.

    Attributes:
        model: Model identifier (e.g. "deepseek-chat", "gpt-4").
        api_key: API key for the LLM provider.
        base_url: Base URL for the API endpoint.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens per response.
        cache_responses: Whether to cache LLM responses.
            **Enforced default True** — override via env var only.
        max_concurrent_calls: Max concurrent LLM calls.
    """

    model: str = "deepseek-chat"
    api_key: SecretStr | None = Field(default=None)
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.2
    max_tokens: int = 4096
    cache_responses: bool = True
    max_concurrent_calls: int = 3

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


class EvaluationConfig(BaseModel):
    """Feature evaluation configuration.

    Attributes:
        cv_folds: Number of cross-validation folds.
        test_size: Fraction of data for test split.
    """

    cv_folds: int = 5
    test_size: float = 0.4

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
        env_file=".env",
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


def get_settings(**overrides) -> Settings:
    """Get settings with optional overrides.

    This is the primary way to obtain configuration in the codebase.
    """
    return Settings(**overrides)
