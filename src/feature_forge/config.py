"""Immutable, validated configuration using pydantic-settings.

Configuration priority (highest to lowest):
1. Constructor arguments
2. Environment variables (FF_* prefix; a local gitignored ``.env`` file is
   loaded via pydantic-settings ``env_file`` — real env vars win over the file)
3. YAML files (config/settings.yaml)

Secrets live in a local, gitignored ``.env`` file (see ``.env.example``) or
exported ``FF_*`` environment variables — never in committed files.
Non-sensitive defaults live in ``config/settings.yaml``.

Example:
    >>> from feature_forge.config import Settings
    >>> settings = Settings()
    >>> settings.task
    'classification'
    >>> settings.llm.model
    'deepseek-chat'
"""

from __future__ import annotations

import functools
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
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
        cache_dir: Directory for the SQLite-backed LLM response cache.
        cache_ttl_days: Optional entry age (in days) after which cached
            responses expire. ``None`` keeps entries forever (historical
            default — caches are reproducibility artifacts).
        cache_size_limit_mb: Optional total cache size cap in MiB; when the
            cap is exceeded, oldest-touched entries are evicted first
            (diskcache LRU culling).
        max_concurrent_calls: Max concurrent LLM calls.
    """

    model: str = "deepseek-chat"
    provider: Literal["auto", "deepseek", "openai", "anthropic", "litellm"] = "auto"
    api_key: SecretStr | None = None
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int = 32768
    agent_max_tokens: int = 8192
    codegen_max_tokens: int = 16384
    cache_responses: bool = True
    cache_dir: str = "memory_files/llm_cache"
    cache_ttl_days: float | None = Field(default=None, gt=0)
    cache_size_limit_mb: float | None = Field(default=None, gt=0)
    max_concurrent_calls: int = 3
    thinking_enabled: bool = False
    reasoning_effort: Literal["low", "medium", "high", "max"] = "medium"

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

    @field_validator("agent_max_tokens", "codegen_max_tokens")
    @classmethod
    def _validate_split_max_tokens(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"token limits must be >= 1, got {v}")
        return v


class TrackerConfig(BaseModel):
    """Experiment tracking configuration.

    Attributes:
        backend: Which tracker to use.
        project: Project name in the tracker.
        entity: Team/entity name (WandB only).
    """

    backend: Literal["wandb", "mlflow", "none"] = "none"
    project: str = "feature-forge"
    entity: str | None = None


class ExecutionEngine(StrEnum):
    """Case execution engine selected by the public platform configuration.

    Hamilton is the sole engine (ADR 0016); the ``legacy`` member was removed
    after qualification. Stale ``legacy`` configuration is rejected with
    actionable migration guidance instead of being reinterpreted.
    """

    HAMILTON = "hamilton"


class HamiltonCacheConfig(BaseModel):
    """Hamilton DAG cache settings.

    Hamilton caching is independent from the mandatory LLM response cache.
    ``path=None`` lets the runtime resolve a cache below ``artifact_root``.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = True
    path: Path | None = None
    log_to_file: bool = False
    max_age_days: float | None = Field(default=None, gt=0)
    max_size_mb: float | None = Field(default=None, gt=0)
    telemetry_max_events: int = Field(default=10_000, ge=1)


_LEGACY_REMOVAL_ADVICE = (
    "Hamilton is the sole execution engine since ADR 0016; the legacy engine "
    "and the 'legacy' artifact policy were removed after qualification and are "
    "never silently reinterpreted. Remove the stale key or set "
    "dataflow.engine to 'hamilton' — see docs/migration_guide.md. Operational "
    "rollback after this release is a downgrade to the last compatibility "
    "release, not a runtime engine switch."
)


class DataflowConfig(BaseModel):
    """Execution and durable-artifact policy for experiment cases.

    Hamilton is the sole execution engine (ADR 0016). Stale ``legacy``
    settings (``engine`` or ``artifact_policy``) are rejected with an
    actionable migration error instead of being silently reinterpreted.
    Medallion layer boundaries (``artifact_policy='layer_boundaries'``) are
    mandatory for every platform run and derived by default.
    """

    model_config = {"extra": "forbid"}

    engine: ExecutionEngine = ExecutionEngine.HAMILTON
    artifact_policy: Literal["layer_boundaries"] | None = None
    artifact_root: Path = Path("experiments/artifacts")
    cache: HamiltonCacheConfig = Field(default_factory=HamiltonCacheConfig)

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_legacy_settings(cls, values: object) -> object:
        """Fail fast on pre-removal legacy settings with migration guidance."""
        if not isinstance(values, dict):
            return values
        stale: list[str] = []
        engine = values.get("engine")
        engine_value = engine.value if isinstance(engine, ExecutionEngine) else engine
        if isinstance(engine_value, str) and engine_value.strip().lower() == "legacy":
            stale.append(
                "dataflow.engine is set to 'legacy' "
                "(YAML `dataflow.engine: legacy` or env `FF_DATAFLOW__ENGINE=legacy`)"
            )
        policy = values.get("artifact_policy")
        if isinstance(policy, str) and policy.strip().lower() == "legacy":
            stale.append(
                "dataflow.artifact_policy is set to 'legacy' "
                "(YAML `dataflow.artifact_policy: legacy` or env "
                "`FF_DATAFLOW__ARTIFACT_POLICY=legacy`)"
            )
        if stale:
            raise ValueError(
                "Rejected stale legacy execution configuration (ADR 0016): "
                + "; ".join(stale)
                + ". "
                + _LEGACY_REMOVAL_ADVICE
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def _derive_policy(cls, values: object) -> object:
        """Derive the mandatory layer-boundaries policy when unset."""
        if isinstance(values, dict) and "artifact_policy" not in values:
            return {**values, "artifact_policy": "layer_boundaries"}
        return values


class FailurePolicy(StrEnum):
    """Scheduler behavior after a terminal failed case result (ADR 0017).

    ``CONTINUE`` keeps scheduling every requested case (historical default).
    ``FAIL_FAST`` stops scheduling new cases after the first terminal failed
    case result; feature-level rejections, warnings, cache misses, and
    recovered retries never trigger it.
    """

    CONTINUE = "continue"
    FAIL_FAST = "fail_fast"


class ExecutionPolicyConfig(BaseModel):
    """Outer experiment scheduler policy.

    Runtime enforcement (sequential fail-fast scheduling, bounded process
    submission, cooperative cancellation) is implemented per plan 22 /
    ADR 0017 (accepted 2026-09-14).
    """

    model_config = {"extra": "forbid"}

    failure_policy: FailurePolicy = FailurePolicy.CONTINUE


class RouterConfig(BaseModel):
    """Router agent configuration.

    Attributes:
        strategy: How to select agents each round.
        max_agents: Maximum agents to activate (None = dynamic).
        min_agents: Minimum agents to activate.
        warmup_rounds: Number of initial rounds where all agents are active.
    """

    strategy: Literal["data_driven", "performance_driven", "hybrid", "llm"] = "hybrid"
    max_agents: int | None = None
    min_agents: int = 1
    warmup_rounds: int = 1

    @field_validator("min_agents")
    @classmethod
    def _validate_min_agents(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"min_agents must be >= 1, got {v}")
        return v

    @field_validator("warmup_rounds")
    @classmethod
    def _validate_warmup_rounds(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"warmup_rounds must be >= 0, got {v}")
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


class SandboxProfile(StrEnum):
    """Operating-system sandbox profile.

    ``STRICT`` is the production profile and requires kernel containment.
    ``DEGRADED_DEVELOPMENT`` is an explicit development-only escape hatch for
    hosts without a qualified containment mechanism; it retains AST and
    process isolation but is never presented as filesystem isolation.
    """

    STRICT = "strict"
    DEGRADED_DEVELOPMENT = "degraded_development"


class EvaluationConfig(BaseModel):
    """Feature evaluation configuration.

    Attributes:
        protocol: Evaluation protocol (ADR 0018). ``holdout`` (default) uses
            deterministic discovery/evaluation partitions so method fitting
            sees discovery rows only; ``compatibility`` reproduces the legacy
            all-row behavior and is only selected explicitly — its results are
            selection-biased and must not be presented as held-out
            performance. (Full selection/reporting partition enforcement
            lands with plan 23 PR 3.)
        evaluation_holdout_fraction: Evaluation-partition fraction under the
            holdout protocol (strictly between 0 and 0.5). Ignored under the
            compatibility protocol.
        cv_folds: Number of cross-validation folds.
        fail_on_feature_error: Raise on feature evaluation failure instead of logging.
        fail_on_agent_error: Raise on agent generation failure instead of skipping.
        sandbox_timeout_seconds: Max seconds for sandbox worker execution.
        sandbox_max_memory_mb: Max address-space (MB) for the sandbox worker.
            Must exceed the pandas/pyarrow/OpenBLAS runtime footprint (~1 GB
            VSZ); 2048 keeps headroom for loaded libraries plus user data.
        sandbox_profile: ``strict`` (the production default) requires Linux
            Landlock containment and fails closed when unavailable;
            ``degraded_development`` must be selected explicitly and is
            recorded as degraded provenance.
        max_candidate_features: Cap on candidate features sent to CV scoring.
        booster_n_jobs: n_jobs for OpenMP-backed estimators (xgboost,
            lightgbm, random_forest) when Intel acceleration is inactive.
            Default 1 (safe pin); 2026-09-06 benchmarks measured 16-234x
            regressions for nested n_jobs>1 fits on a busy shared host, so
            only raise this on a quiet, measured machine. -1 = all cores.
            Forced to 1 whenever Intel acceleration is active (ADR 0010).
    """

    protocol: Literal["holdout", "compatibility"] = "holdout"
    evaluation_holdout_fraction: float = Field(default=0.25, gt=0.0, lt=0.5)
    cv_folds: int = 5
    fail_on_feature_error: bool = False
    fail_on_agent_error: bool = False
    sandbox_timeout_seconds: float = 5.0
    sandbox_max_memory_mb: int = 2048
    sandbox_profile: SandboxProfile = SandboxProfile.STRICT
    max_candidate_features: int = 50
    max_cv_workers: int | None = None
    feature_eval_backend: Literal["threading", "loky"] = "threading"
    booster_n_jobs: int = 1

    @field_validator("booster_n_jobs")
    @classmethod
    def _validate_booster_n_jobs(cls, v: int) -> int:
        """Allow -1 (all cores) or any positive count; anything else is a typo."""
        if v != -1 and v < 1:
            raise ValueError("booster_n_jobs must be -1 (all cores) or >= 1")
        return v

    @field_validator("cv_folds")
    @classmethod
    def _validate_cv_folds(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"cv_folds must be >= 2, got {v}")
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

    @field_validator("max_cv_workers")
    @classmethod
    def _validate_max_cv_workers(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"max_cv_workers must be >= 1 when set, got {v}")
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
        env_file=".env",
        extra="ignore",
    )

    # Core pipeline settings
    task: Literal["classification", "regression"] = "classification"
    metric: str = "auc"
    n_rounds: int = 4
    max_selected_features: int = Field(default=2, ge=1)
    random_state: int = 42
    # Intel Extension for Scikit-learn (sklearnex) acceleration. When True and
    # scikit-learn-intelex is installed (the `intel` extra), sklearn is patched
    # to run on the Intel/DAAL backend (libiomp5). Boosting estimators stay at
    # n_jobs=1 and evaluation stays in-process (threading joblib backend, no
    # fork-after-OpenMP) to avoid the libiomp5/libgomp OpenMP deadlock with the
    # libgomp-linked XGBoost/LightGBM wheels. See
    # docs/decisions/0010-intel-openmp-bootstrap.md.
    intel_acceleration: bool = True

    # Subsystem configs
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    dataflow: DataflowConfig = Field(default_factory=DataflowConfig)
    execution: ExecutionPolicyConfig = Field(default_factory=ExecutionPolicyConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_min_effective(cls, values: object) -> object:
        """Accept the old constructor/YAML key without changing env prefixes."""
        if isinstance(values, dict) and "min_effective" in values:
            migrated = dict(values)
            migrated["max_selected_features"] = migrated.pop("min_effective")
            return migrated
        return values

    @field_validator("n_rounds")
    @classmethod
    def _validate_n_rounds(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n_rounds must be >= 1, got {v}")
        return v

    @field_validator("metric")
    @classmethod
    def _validate_metric(cls, v: str) -> str:
        valid = {"auc", "acc", "f1", "rmse", "mae", "r2", "nrmse"}
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


@functools.lru_cache(maxsize=1)
def _get_base_settings() -> Settings:
    return Settings()


def get_settings(*, invalidate: bool = False, **overrides: Any) -> Settings:
    if invalidate:
        _get_base_settings.cache_clear()
    if not overrides:
        return _get_base_settings()
    return Settings(**overrides)
