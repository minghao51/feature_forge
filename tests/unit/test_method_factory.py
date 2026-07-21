"""Tests for explicit method construction via ``from_run_context``.

Covers P0-4 from `.claude/handoffs/2026-07-20-remote-cleanup.md`: the platform
must inject an LLM client and case-resolved settings/evaluator/mode into every
method adapter *without* relying on ``inspect.signature`` reflection. The
historical gaps closed here:

* CAAFE unified raised through the platform used to receive ``llm_client=None``
  and fail at fit time with ``EvaluationError("llm_client required...")``.
* MALMAS silently dropped ``case.mode`` because it was not part of
  ``MALMASMethod.__init__``'s signature.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from feature_forge.config import Settings
from feature_forge.evaluation import CVEvaluator
from feature_forge.evaluation.kit import EvaluationKit
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.experiment.context import CaseExecutionContext, resolve_case_context
from feature_forge.experiment.execution import ExperimentCase
from feature_forge.methods.caafe.method import CAAFEMethod
from feature_forge.methods.llmfe.method import LLMFEMethod
from feature_forge.methods.malmas.method import MALMASMethod
from feature_forge.methods.malmus.method import MalmusMethod
from feature_forge.methods.openfe.method import OpenFEMethod


def _make_context(
    *,
    method: str = "caafe",
    mode: str | None = None,
    metric: str | None = None,
    task: str = "classification",
    llm_client: Any = None,
    seed: int = 42,
) -> CaseExecutionContext:
    """Build a CaseExecutionContext with a stub LLM client (no API calls)."""
    settings = Settings(task=task, metric=metric or ("auc" if task == "classification" else "r2"))
    settings.random_state = seed
    case = ExperimentCase(
        dataset="dummy",
        method=method,
        model="random_forest",
        seed=seed,
        mode=mode,
        metric=metric,
    )
    loaded = {
        "target": "y",
        "train": pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}),
        "metadata": {"task": task},
    }
    ctx = resolve_case_context(
        case=case,
        base_settings=settings,
        loaded_dataset=loaded,
        registry_metadata={},
    )
    # Override the best-effort llm_client (which is None without an API key)
    # with an explicit stub so we can assert it is forwarded into methods.
    return CaseExecutionContext(
        case=ctx.case,
        task=ctx.task,
        metric=ctx.metric,
        seed=ctx.seed,
        cv_folds=ctx.cv_folds,
        settings=ctx.settings,
        evaluation=ctx.evaluation,
        model_factory=ctx.model_factory,
        evaluator=ctx.evaluator,
        run_id=ctx.run_id,
        case_fingerprint=ctx.case_fingerprint,
        execution_profile=ctx.execution_profile,
        artifact_policy=ctx.artifact_policy,
        llm_client=llm_client,
        sandbox=ctx.sandbox,
        eval_kit=ctx.eval_kit,
    )


class _StubLLMClient:
    """Minimal LLM client stub — identity marker for forwarding assertions."""


class TestFromRunContextWiring:
    """Each built-in method's ``from_run_context`` must inject the right deps."""

    def test_caafe_receives_context_llm_client(self):
        client = _StubLLMClient()
        ctx = _make_context(method="caafe", mode="unified", llm_client=client)
        method = CAAFEMethod.from_run_context(ctx)
        assert method.llm_client is client
        assert method.variant == "unified"
        # The case-resolved evaluator is reused (not rebuilt).
        assert method.evaluator is ctx.evaluator

    def test_caafe_mode_fidelity_routes_to_fidelity_variant(self):
        ctx = _make_context(method="caafe", mode="fidelity", llm_client=None)
        method = CAAFEMethod.from_run_context(ctx)
        assert method.variant == "fidelity"

    def test_caafe_default_mode_is_unified(self):
        ctx = _make_context(method="caafe", mode=None, llm_client=None)
        method = CAAFEMethod.from_run_context(ctx)
        assert method.variant == "unified"

    def test_llmfe_receives_context_llm_client_and_mode(self):
        client = _StubLLMClient()
        ctx = _make_context(method="llmfe", mode="iterative", llm_client=client)
        method = LLMFEMethod.from_run_context(ctx)
        assert method.llm_client is client
        assert method.mode == "iterative"
        assert method.evaluator is ctx.evaluator
        # Settings are case-resolved, not global.
        assert method.llm_client is not None  # type: ignore[unreachable]

    def test_llmfe_default_mode_is_single_shot(self):
        # Provide a stub client so the method does not attempt to self-build
        # a real provider (which would require an API key).
        ctx = _make_context(method="llmfe", mode=None, llm_client=_StubLLMClient())
        method = LLMFEMethod.from_run_context(ctx)
        assert method.mode == "single_shot"

    def test_malmus_receives_context_llm_client_and_mode(self):
        client = _StubLLMClient()
        ctx = _make_context(method="malmus", mode="iterative", llm_client=client)
        method = MalmusMethod.from_run_context(ctx)
        assert method.llm_client is client
        assert method.mode == "iterative"

    def test_openfe_does_not_consume_llm_client(self):
        client = _StubLLMClient()
        ctx = _make_context(
            method="openfe",
            mode=None,
            metric="auc",
            llm_client=client,
        )
        method = OpenFEMethod.from_run_context(ctx)
        # OpenFE has no llm_client attribute — it is the non-LLM baseline.
        assert not hasattr(method, "llm_client")
        assert method.metric == "auc"
        assert method.n_jobs >= 1

    def test_openfe_n_jobs_follows_resource_setting(self):
        settings = Settings(evaluation={"max_cv_workers": 3})
        case = ExperimentCase(dataset="d", method="openfe", model="xgboost", seed=1)
        ctx = resolve_case_context(
            case=case,
            base_settings=settings,
            loaded_dataset={
                "target": "y",
                "train": pd.DataFrame({"x": [1, 2], "y": [0, 1]}),
                "metadata": {"task": "classification"},
            },
            registry_metadata={},
        )
        method = OpenFEMethod.from_run_context(ctx)
        assert method.n_jobs == 3

    def test_malmas_forwards_mode_and_llm_client_into_kwargs(self):
        client = _StubLLMClient()
        ctx = _make_context(method="malmas", mode="no_memory", llm_client=client)
        method = MALMASMethod.from_run_context(ctx)
        # mode + llm_client are captured for FeatureForge construction.
        assert method._kwargs.get("mode") == "no_memory"
        assert method._kwargs.get("llm_client") is client
        # The case-resolved settings are forwarded as `config`.
        assert method._config is ctx.settings

    def test_malmas_default_mode_is_full(self):
        ctx = _make_context(method="malmas", mode=None, llm_client=None)
        method = MALMASMethod.from_run_context(ctx)
        assert method._kwargs.get("mode") == "full"


class TestMALMASModeReachesFeatureForge:
    """Regression: ``case.mode`` must reach ``FeatureForge.mode`` at fit time.

    Historically ``MALMASMethod.__init__`` only accepted ``config`` and
    ``**kwargs``, and the ``inspect.signature``-based construction in
    ``CaseComputation._method_kwargs`` never injected ``mode`` — so selecting
    an ablation pipeline (e.g. ``no_memory``) through the platform silently
    ran the full pipeline instead.
    """

    def test_no_memory_mode_propagates(self, monkeypatch):
        captured: dict[str, Any] = {}

        class _StubFeatureForge:
            def __init__(self_inner, **kwargs: Any) -> None:
                captured.update(kwargs)

            def fit(self_inner, X, y, **kwargs):
                return self_inner

            def transform(self_inner, X):
                return X

            def close(self_inner) -> None:
                pass

            generated_scripts: ClassVar[list[str]] = []

            @property
            def feature_metadata(self_inner) -> list[dict[str, Any]]:
                return []

            def get_artifacts(self_inner) -> dict[str, Any]:
                return {}

        import feature_forge.methods.malmas.method as malmas_module

        monkeypatch.setattr(malmas_module, "FeatureForge", _StubFeatureForge)

        client = _StubLLMClient()
        ctx = _make_context(method="malmas", mode="no_memory", llm_client=client)
        method = MALMASMethod.from_run_context(ctx)
        method.fit(pd.DataFrame({"x": [1, 2]}), pd.Series([0, 1]))

        assert captured.get("mode") == "no_memory"
        assert captured.get("llm_client") is client


class TestCaseContextLlmClientConstruction:
    """``resolve_case_context`` builds a best-effort client or None."""

    def test_context_carries_llm_client_sandbox_and_eval_kit(self):
        # Without an API key the client is None, but the context still carries
        # a sandbox and eval_kit for non-LLM methods.
        settings = Settings(task="classification", metric="auc")
        case = ExperimentCase(dataset="d", method="openfe", model="xgboost", seed=1)
        ctx = resolve_case_context(
            case=case,
            base_settings=settings,
            loaded_dataset={
                "target": "y",
                "train": pd.DataFrame({"x": [1, 2], "y": [0, 1]}),
                "metadata": {"task": "classification"},
            },
            registry_metadata={},
        )
        assert isinstance(ctx.sandbox, SandboxedExecutor)
        assert isinstance(ctx.eval_kit, EvaluationKit)
        assert ctx.eval_kit.sandbox is ctx.sandbox
        assert ctx.eval_kit.evaluator is ctx.evaluator
        # No API key configured → best-effort client is None (not an error).
        assert ctx.llm_client is None

    def test_context_evaluator_metric_direction_matches_resolved_metric(self):
        settings = Settings(task="regression", metric="rmse")
        case = ExperimentCase(dataset="d", method="llmfe", model="xgboost", seed=1)
        ctx = resolve_case_context(
            case=case,
            base_settings=settings,
            loaded_dataset={
                "target": "y",
                "train": pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}),
                "metadata": {"task": "regression"},
            },
            registry_metadata={},
        )
        from feature_forge.evaluation.metrics import MetricDirection

        assert ctx.metric == "rmse"
        assert ctx.evaluator.metric_direction is MetricDirection.MINIMIZE


class TestConstructMethodDispatch:
    """``CaseComputation._construct_method`` prefers ``from_run_context``."""

    def test_dispatch_uses_from_run_context_when_present(self):
        from feature_forge.experiment.case_executor import CaseComputation

        client = _StubLLMClient()
        ctx = _make_context(method="caafe", mode="unified", llm_client=client)
        method = CaseComputation._construct_method(CAAFEMethod, ctx)
        assert isinstance(method, CAAFEMethod)
        assert method.llm_client is client

    def test_dispatch_falls_back_to_method_kwargs_for_plugins(self):
        from feature_forge.experiment.case_executor import CaseComputation

        # A third-party-style plugin that does NOT implement from_run_context
        # but accepts `settings`/`evaluator` via the legacy reflection path.
        class _LegacyPlugin:
            def __init__(self, settings: Settings, evaluator: CVEvaluator, **kw: Any) -> None:
                self.settings = settings
                self.evaluator = evaluator
                self.kwargs = kw

        ctx = _make_context(method="legacy", mode=None, llm_client=None)
        method = CaseComputation._construct_method(_LegacyPlugin, ctx)  # type: ignore[arg-type]
        assert method.settings is ctx.settings  # type: ignore[attr-defined]
        assert method.evaluator is ctx.evaluator  # type: ignore[attr-defined]
