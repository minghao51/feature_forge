"""Core pipeline for single-round feature engineering.

Orchestrates agents, code generation, sandboxed execution,
and feature evaluation.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd
from joblib import Parallel, delayed  # type: ignore[import-untyped]

from feature_forge.config import Settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.kit import EvaluationKit
from feature_forge.evaluation.metrics import MetricDirection, get_metric_direction
from feature_forge.evaluation.prefilter import prefilter_candidate_columns
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import CodeExecutionError, PipelineError
from feature_forge.llm.base import LLMClient
from feature_forge.methods.malmas.agents.base import Agent
from feature_forge.methods.malmas.pipeline.codegen import CodeGenerator
from feature_forge.methods.malmas.pipeline.result import PipelineResult
from feature_forge.observability.structlog_config import get_logger
from feature_forge.runtime.intel_bootstrap import is_intel_active
from feature_forge.storage.hashing import fingerprint
from feature_forge.types import FeatureSpec

logger = get_logger(__name__)


async def _exec_sandbox(
    sandbox: SandboxedExecutor,
    agent_name: str,
    code: str,
    X: pd.DataFrame,
    source: str,
    sandbox_timeout: float,
) -> tuple[str, pd.DataFrame] | None:
    t0 = time.perf_counter()
    try:
        part = await asyncio.wait_for(
            asyncio.to_thread(
                sandbox.execute,
                code,
                X,
                source=source,
                agent_name=agent_name,
            ),
            timeout=max(sandbox_timeout * 2, 30.0),
        )
        event = "agent_sandbox_complete" if "test" not in source else "agent_sandbox_complete_test"
        logger.info(
            event,
            agent=agent_name,
            result_shape=part.shape,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return (agent_name, part)
    except TimeoutError:
        logger.warning(
            "agent_sandbox_timeout",
            agent=agent_name,
            timeout=sandbox_timeout * 2,
        )
        return None
    except CodeExecutionError as exc:
        logger.warning(
            "agent_code_execution_failed",
            agent=agent_name,
            error=str(exc)[:200],
        )
        return None
    except Exception as exc:
        logger.exception(
            "agent_code_execution_unexpected",
            agent=agent_name,
            error=str(exc)[:200],
        )
        return None


@dataclass(frozen=True)
class _SuccessfulCode:
    """One unique code batch that succeeded on training data."""

    agent_name: str
    code: str
    feature_names: tuple[str, ...]
    dtypes: tuple[str, ...]


class CorePipeline:
    """Single-round feature engineering pipeline.

    Steps:
    1. Run selected agents to get feature specs
    2. Generate code for specs
    3. Execute code in sandbox
    4. Evaluate features via CV
    5. Return top features
    """

    def __init__(
        self,
        config: Settings,
        llm_client: LLMClient,
        eval_kit: EvaluationKit | None = None,
        evaluator: CVEvaluator | None = None,
        sandbox: SandboxedExecutor | None = None,
        code_generator: CodeGenerator | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.eval_kit: EvaluationKit | None = eval_kit
        if eval_kit is not None:
            self.evaluator = eval_kit.evaluator
            self.sandbox = eval_kit.sandbox
        else:
            self.evaluator = evaluator or CVEvaluator(config)
            self.sandbox = sandbox or SandboxedExecutor(
                timeout_seconds=config.evaluation.sandbox_timeout_seconds,
                max_memory_mb=config.evaluation.sandbox_max_memory_mb,
            )
        self.code_generator = code_generator or CodeGenerator(
            llm_client, max_tokens=config.llm.codegen_max_tokens
        )
        self._baseline_cache: dict[str, float] = {}

    def _baseline_cache_key(self, X_train: pd.DataFrame, y_train: pd.Series) -> str:
        """Return canonical data, fold, metric, and model identity."""
        return fingerprint(
            {
                "kind": "malmas-baseline",
                "features": {
                    "columns": [str(column) for column in X_train.columns],
                    "dtypes": [str(dtype) for dtype in X_train.dtypes],
                    "values": [
                        int(value)
                        for value in pd.util.hash_pandas_object(X_train, index=True).array
                    ],
                },
                "target": {
                    "name": str(y_train.name),
                    "dtype": str(y_train.dtype),
                    "values": [
                        int(value)
                        for value in pd.util.hash_pandas_object(y_train, index=True).array
                    ],
                },
                "evaluation": {
                    "task": self.config.task,
                    "metric": self.config.metric,
                    "cv_folds": getattr(
                        self.evaluator, "cv_folds", self.config.evaluation.cv_folds
                    ),
                    "random_state": self.config.random_state,
                    "model": getattr(self.evaluator, "default_model_name", "random_forest"),
                },
            }
        )

    def _is_improvement(self, gain: float) -> bool:
        check = getattr(self.evaluator, "is_improvement", None)
        if callable(check):
            return bool(check(gain))
        direction = get_metric_direction(self.config.metric)
        return gain < 0 if direction is MetricDirection.MINIMIZE else gain > 0

    def _improvement_value(self, gain: float) -> float:
        normalize = getattr(self.evaluator, "improvement_value", None)
        if callable(normalize):
            return float(normalize(gain))
        return (
            -gain if get_metric_direction(self.config.metric) is MetricDirection.MINIMIZE else gain
        )

    async def run(
        self,
        agents: list[Agent],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame | None = None,
        context: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> PipelineResult:
        """Run one round of feature engineering.

        Returns:
            Dict with keys:
            - features_train: DataFrame of generated features for train
            - features_test: DataFrame of generated features for test
            - agent_gains: Dict[str, DataFrame] per-agent feature gains
            - specs: List[FeatureSpec] all generated specs
            - top_features: DataFrame of top-k effective features
            - generated_code: Python code string executed in sandbox
        """
        pipeline_t0 = time.perf_counter()
        logger.info(
            "pipeline_start",
            agents=[a.name for a in agents],
            num_agents=len(agents),
            train_shape=X_train.shape,
        )

        per_agent_contexts: list[dict[str, Any]] = (
            context if isinstance(context, list) else [dict(context or {}) for _ in agents]
        )
        semaphore = asyncio.Semaphore(self.config.llm.max_concurrent_calls)

        all_specs = await self._generate_specs(
            agents, X_train, y_train, per_agent_contexts, semaphore
        )
        if not all_specs:
            logger.info(
                "pipeline_complete", num_specs=0, num_selected=0, reason="no_specs_generated"
            )
            return self._empty_result(X_train, X_test)

        schema = self._build_schema(X_train)
        features_train, successful_code, feature_failures = await self._execute_train(
            all_specs, X_train, schema, semaphore
        )

        features_test, test_failures = await self._execute_test(successful_code, X_test)
        feature_failures.extend(test_failures)
        code = "\n\n".join(batch.code for batch in successful_code)

        features_train = features_train.reindex(X_train.index)
        if X_test is not None:
            features_test = features_test.reindex(X_test.index)

        result = self._evaluate_and_select(
            features_train,
            features_test,
            X_train,
            y_train,
            X_test,
            all_specs,
            agents,
            code,
            successful_code,
            feature_failures,
        )
        pipeline_latency_ms = round((time.perf_counter() - pipeline_t0) * 1000, 1)
        logger.info(
            "pipeline_complete",
            num_specs=len(all_specs),
            num_selected=len(result.selected_features_train.columns)
            if not result.selected_features_train.empty
            else 0,
            num_effective=len([g for g in result.gains.values() if self._is_improvement(g)]),
            baseline_score=round(result.baseline_score, 6),
            latency_ms=pipeline_latency_ms,
        )
        return result

    @staticmethod
    def _empty_result(X_train: pd.DataFrame, X_test: pd.DataFrame | None) -> PipelineResult:
        empty_train = pd.DataFrame(index=X_train.index)
        empty_test = pd.DataFrame(index=X_test.index if X_test is not None else X_train.index)
        return PipelineResult(
            features_train=empty_train,
            features_test=empty_test,
            all_features_train=empty_train,
            all_features_test=empty_test,
            selected_features_train=empty_train,
            selected_features_test=empty_test,
            top_features_train=empty_train,
            top_features_test=empty_test,
            agent_gains={},
            specs=[],
            baseline_score=0.0,
            gains={},
            generated_code="",
            generated_codes=[],
            code_features=[],
            feature_failures=[],
        )

    @staticmethod
    def _build_schema(X_train: pd.DataFrame) -> dict[str, Any]:
        return {
            "shape": list(X_train.shape),
            "columns": {
                col: {
                    "dtype": str(X_train[col].dtype),
                    "nullable": bool(X_train[col].isna().any()),
                }
                for col in X_train.columns
            },
            "index_type": type(X_train.index).__name__,
            "index_length": len(X_train),
        }

    async def _generate_specs(
        self,
        agents: list[Agent],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        per_agent_contexts: list[dict[str, Any]],
        semaphore: asyncio.Semaphore,
    ) -> list[FeatureSpec]:
        async def _run_agent(agent: Agent, ctx: dict[str, Any]) -> list[FeatureSpec]:
            async with semaphore:
                return await agent.generate(X_train, y_train, ctx)

        agent_specs_list = await asyncio.gather(
            *[_run_agent(a, c) for a, c in zip(agents, per_agent_contexts, strict=False)],
            return_exceptions=True,
        )

        all_specs: list[FeatureSpec] = []
        for agent, specs in zip(agents, agent_specs_list, strict=False):
            if isinstance(specs, BaseException):
                logger.warning(
                    "agent_generation_failed",
                    agent=agent.name,
                    error=str(specs),
                )
                if self.config.evaluation.fail_on_agent_error:
                    raise PipelineError(f"Agent '{agent.name}' failed: {specs}") from specs
                continue
            normalized: list[FeatureSpec] = []
            for spec in specs:
                if isinstance(spec, dict):
                    spec_obj = FeatureSpec(**spec)
                else:
                    spec_obj = spec
                spec_obj.agent_name = agent.name
                normalized.append(spec_obj)
            all_specs.extend(normalized)
            logger.info("agent_specs_generated", agent=agent.name, num_specs=len(specs))
        return all_specs

    async def _execute_train(
        self,
        all_specs: list[FeatureSpec],
        X_train: pd.DataFrame,
        schema: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> tuple[pd.DataFrame, list[_SuccessfulCode], list[dict[str, str]]]:
        specs_by_agent: dict[str, list[FeatureSpec]] = defaultdict(list)
        for spec in all_specs:
            specs_by_agent[spec.agent_name].append(spec)

        logger.info(
            "code_generation_start", num_specs=len(all_specs), num_agents=len(specs_by_agent)
        )
        code_gen_t0 = time.perf_counter()

        async def _gen_for_agent(
            agent_name: str, agent_specs: list[FeatureSpec]
        ) -> tuple[str, str] | None:
            last_error: str | None = None
            for attempt in range(2):
                try:
                    async with semaphore:
                        code = await self.code_generator.generate_code(
                            agent_specs, schema=schema, error_feedback=last_error
                        )
                    return (agent_name, code)
                except Exception as exc:
                    last_error = str(exc)
                    logger.warning(
                        "agent_code_gen_failed",
                        agent=agent_name,
                        attempt=attempt,
                        error=last_error[:200],
                    )
            return None

        code_gen_results = await asyncio.gather(
            *[_gen_for_agent(name, specs) for name, specs in specs_by_agent.items()]
        )
        generated = [result for result in code_gen_results if result is not None]

        # Group identical code before execution. Expected feature names are
        # unioned so duplicate batches execute exactly once without losing
        # schema accountability for either agent.
        grouped: dict[str, dict[str, Any]] = {}
        for agent_name, code in generated:
            batch = grouped.setdefault(code, {"agent": agent_name, "features": []})
            known = set(batch["features"])
            for spec in specs_by_agent[agent_name]:
                if spec.name not in known:
                    batch["features"].append(spec.name)
                    known.add(spec.name)
        unique_code_parts = [
            (str(batch["agent"]), code, tuple(str(name) for name in batch["features"]))
            for code, batch in grouped.items()
        ]

        sandbox_timeout = self.config.evaluation.sandbox_timeout_seconds

        exec_results = await asyncio.gather(
            *[
                _exec_sandbox(
                    self.sandbox, name, code, X_train, "malmas_core_train", sandbox_timeout
                )
                for name, code, _feature_names in unique_code_parts
            ]
        )

        features_train_parts: list[pd.DataFrame] = []
        successful: list[_SuccessfulCode] = []
        failures: list[dict[str, str]] = []
        for (_name, code, expected_names), result in zip(
            unique_code_parts, exec_results, strict=True
        ):
            if result is None:
                failures.extend(
                    {
                        "feature": name,
                        "phase": "train",
                        "reason_code": "execution_failed",
                    }
                    for name in expected_names
                )
                continue
            output = result[1]
            if not output.index.equals(X_train.index):
                failures.extend(
                    {
                        "feature": name,
                        "phase": "train",
                        "reason_code": "index_mismatch",
                    }
                    for name in expected_names
                )
                continue
            valid_names: list[str] = []
            for name in expected_names:
                if name not in output.columns:
                    failures.append(
                        {
                            "feature": name,
                            "phase": "train",
                            "reason_code": "missing_output",
                        }
                    )
                else:
                    valid_names.append(name)
            for name in output.columns:
                if name not in expected_names:
                    failures.append(
                        {
                            "feature": str(name),
                            "phase": "train",
                            "reason_code": "unexpected_output",
                        }
                    )
            if not valid_names:
                continue
            valid_output = output[valid_names]
            features_train_parts.append(valid_output)
            successful.append(
                _SuccessfulCode(
                    agent_name=_name,
                    code=code,
                    feature_names=tuple(valid_names),
                    dtypes=tuple(str(valid_output[name].dtype) for name in valid_names),
                )
            )

        if not features_train_parts:
            raise PipelineError("All agent code executions failed schema validation")

        features_train = self._concat_dedup(features_train_parts)

        logger.info(
            "code_generation_complete",
            num_specs=len(all_specs),
            num_agents=len(specs_by_agent),
            code_length=sum(len(batch.code) for batch in successful),
            num_failures=len(failures),
            latency_ms=round((time.perf_counter() - code_gen_t0) * 1000, 1),
        )
        return features_train, successful, failures

    async def _execute_test(
        self,
        successful_code: list[_SuccessfulCode],
        X_test: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, list[dict[str, str]]]:
        if X_test is None:
            return pd.DataFrame(), []

        sandbox_timeout = self.config.evaluation.sandbox_timeout_seconds
        test_exec_results = await asyncio.gather(
            *[
                _exec_sandbox(
                    self.sandbox,
                    batch.agent_name,
                    batch.code,
                    X_test,
                    "malmas_core_test",
                    sandbox_timeout,
                )
                for batch in successful_code
            ],
            return_exceptions=True,
        )

        features_test_parts: list[pd.DataFrame] = []
        failures: list[dict[str, str]] = []
        for batch, test_result in zip(successful_code, test_exec_results, strict=True):
            if isinstance(test_result, BaseException) or test_result is None:
                failures.extend(
                    {
                        "feature": name,
                        "phase": "test",
                        "reason_code": "execution_failed",
                    }
                    for name in batch.feature_names
                )
                continue
            output = test_result[1]
            if not output.index.equals(X_test.index):
                failures.extend(
                    {
                        "feature": name,
                        "phase": "test",
                        "reason_code": "index_mismatch",
                    }
                    for name in batch.feature_names
                )
                continue
            valid_names: list[str] = []
            for name, train_dtype in zip(batch.feature_names, batch.dtypes, strict=True):
                if name not in output.columns:
                    failures.append(
                        {
                            "feature": name,
                            "phase": "test",
                            "reason_code": "missing_output",
                        }
                    )
                elif str(output[name].dtype) != train_dtype:
                    failures.append(
                        {
                            "feature": name,
                            "phase": "test",
                            "reason_code": "dtype_mismatch",
                        }
                    )
                else:
                    valid_names.append(name)
            if valid_names:
                features_test_parts.append(output[valid_names])
        return self._concat_dedup(features_test_parts, index=X_test.index), failures

    def _evaluate_and_select(
        self,
        features_train: pd.DataFrame,
        features_test: pd.DataFrame,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame | None,
        all_specs: list[FeatureSpec],
        agents: list[Agent],
        code: str,
        successful_code: list[_SuccessfulCode] | None = None,
        feature_failures: list[dict[str, str]] | None = None,
    ) -> PipelineResult:
        _cache_key = self._baseline_cache_key(X_train, y_train)
        baseline_score = self._baseline_cache.get(_cache_key)
        if baseline_score is None:
            baseline_score = self.evaluator.evaluate_baseline(X_train, y_train)
            self._baseline_cache[_cache_key] = baseline_score
        logger.info(
            "evaluation_baseline", score=round(baseline_score, 6), metric=self.config.metric
        )
        gains: dict[str, float] = {}
        candidate_columns = self._prefilter_candidate_columns(features_train)

        max_workers = self.config.evaluation.max_cv_workers
        n_jobs = (
            (max_workers if max_workers is not None else min(os.cpu_count() or 4, 8))
            if len(candidate_columns) > 1
            else 1
        )
        backend = self.config.evaluation.feature_eval_backend
        # When Intel acceleration is actually engaged, sklearnex loads libiomp5
        # while the XGBoost/LightGBM wheels use libgomp. Forking after OpenMP is
        # initialized (the loky/multiprocessing backends) can deadlock in the
        # child process, so we keep evaluation in the *same* process (threading)
        # and rely on the model factory pinning boosters to n_jobs=1 (libgomp
        # stays idle). This avoids the fork-after-OpenMP hazard entirely. See
        # docs/decisions/0010-intel-openmp-bootstrap.md.
        if is_intel_active() and backend != "threading":
            logger.warning(
                "feature_eval_backend_forced_threading",
                configured_backend=backend,
                reason="Intel acceleration active: fork-after-OpenMP (loky) can deadlock; see ADR 0010",
            )
            backend = "threading"
        if (
            len(candidate_columns) > 1
            and backend == "loky"
            and not is_intel_active()
            and X_train.shape[0] * X_train.shape[1] > 2_000_000
        ):
            logger.warning(
                "feature_eval_backend_loky_large_matrix_fallback",
                rows=X_train.shape[0],
                cols=X_train.shape[1],
                candidates=len(candidate_columns),
                hint="Falling back to threading backend to prevent OOM",
            )
            backend = "threading"

        eval_results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(self._eval_single_feature)(
                self.evaluator, X_train, y_train, features_train[[col]], col, baseline_score
            )
            for col in candidate_columns
        )
        for col, result in zip(candidate_columns, eval_results, strict=True):
            if isinstance(result, Exception):
                logger.warning("feature_evaluation_failed", feature=col, error=str(result))
                if self.config.evaluation.fail_on_feature_error:
                    raise PipelineError(
                        f"Feature evaluation failed for '{col}': {result}"
                    ) from result
                gains[col] = (
                    float("inf")
                    if get_metric_direction(self.config.metric) is MetricDirection.MINIMIZE
                    else float("-inf")
                )
            else:
                gains[col] = result
                logger.debug(
                    "feature_evaluated",
                    feature=col,
                    gain=round(result, 6),
                    effective=self._is_improvement(result),
                )

        effective = {k: v for k, v in gains.items() if self._is_improvement(v)}
        top_k = sorted(
            effective.items(),
            key=lambda item: self._improvement_value(item[1]),
            reverse=True,
        )
        top_k_names = [name for name, _ in top_k[: self.config.max_selected_features]]
        if X_test is not None:
            top_k_names = [name for name in top_k_names if name in features_test.columns]

        top_features_train = (
            features_train[top_k_names] if top_k_names else pd.DataFrame(index=X_train.index)
        )
        top_features_test = (
            features_test[top_k_names] if top_k_names and X_test is not None else pd.DataFrame()
        )

        agent_gains: dict[str, pd.DataFrame] = {}
        for agent in agents:
            agent_feature_names = [s.name for s in all_specs if s.agent_name == agent.name]
            agent_gain_rows = []
            for fname in agent_feature_names:
                if fname in gains:
                    agent_gain_rows.append({"feature": fname, "gain": gains[fname]})
            if agent_gain_rows:
                agent_gains[agent.name] = pd.DataFrame(agent_gain_rows)

        return PipelineResult(
            features_train=features_train,
            features_test=features_test,
            all_features_train=features_train,
            all_features_test=features_test if X_test is not None else pd.DataFrame(),
            selected_features_train=top_features_train,
            selected_features_test=top_features_test,
            agent_gains=agent_gains,
            specs=all_specs,
            top_features_train=top_features_train,
            top_features_test=top_features_test,
            baseline_score=baseline_score,
            gains=gains,
            generated_code=code,
            generated_codes=[batch.code for batch in successful_code or []],
            code_features=[list(batch.feature_names) for batch in successful_code or []],
            feature_failures=list(feature_failures or []),
        )

    @staticmethod
    def _concat_dedup(parts: list[pd.DataFrame], index: pd.Index | None = None) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(index=index) if index is not None else pd.DataFrame()
        combined = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]
        dup_cols = combined.columns[combined.columns.duplicated()].tolist()
        if dup_cols:
            logger.warning("column_dedup", duplicated_columns=dup_cols)
        return combined.loc[:, ~combined.columns.duplicated()]

    def _prefilter_candidate_columns(self, features_train: pd.DataFrame) -> list[str]:
        return prefilter_candidate_columns(
            features_train,
            self.config.evaluation.max_candidate_features,
        )

    @staticmethod
    def _eval_single_feature(
        evaluator: CVEvaluator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_df: pd.DataFrame,
        col: str,
        baseline_score: float,
    ) -> float | Exception:
        try:
            return evaluator.evaluate_feature(
                X_train, y_train, feature_df, baseline_score=baseline_score
            )
        except Exception as exc:
            return exc
