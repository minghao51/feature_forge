"""Core pipeline for single-round feature engineering.

Orchestrates agents, code generation, sandboxed execution,
and feature evaluation.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any

import pandas as pd
from joblib import Parallel, delayed  # type: ignore[import-untyped]

from feature_forge.config import Settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import PipelineError
from feature_forge.llm.base import LLMClient
from feature_forge.methods.malmas.agents.base import Agent
from feature_forge.methods.malmas.prompts import get_registry
from feature_forge.observability.structlog_config import get_logger
from feature_forge.types import FeatureSpec
from feature_forge.utils import strip_markdown_fences

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
    except Exception as exc:
        logger.warning(
            "agent_code_execution_failed",
            agent=agent_name,
            error=str(exc)[:200],
        )
        return None


class CodeGenerator:
    """Generates pandas code from feature specifications."""

    def __init__(self, llm_client: LLMClient, max_tokens: int = 32768) -> None:
        self.llm_client = llm_client
        self._system_prompt = get_registry().get("code_generation").system
        self._max_tokens = max_tokens

    async def generate_code(
        self,
        specs: list[FeatureSpec],
        schema: dict[str, Any] | None = None,
        error_feedback: str | None = None,
    ) -> str:
        """Generate Python code for a list of feature specs.

        Includes AST validation and a single retry on syntax errors.
        """
        specs_dump = [s.model_dump() if hasattr(s, "model_dump") else s for s in specs]
        specs_json = json.dumps(specs_dump, indent=2, ensure_ascii=False)

        parts: list[str] = []
        if schema:
            schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
            parts.append(f"Data schema:\n{schema_json}")
        parts.append(f"Generate code for features:\n{specs_json}")
        user_prompt = "\n\n".join(parts)

        if error_feedback:
            user_prompt += f"\n\n{error_feedback}"

        last_error: str | None = None
        for attempt in range(2):
            response = await self.llm_client.complete(
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=self._max_tokens,
            )
            code = strip_markdown_fences(response.content)

            error_msg = _validate_code_ast(code)
            if error_msg is None:
                return code
            last_error = error_msg

            if attempt == 0:
                logger.warning("code_gen_ast_invalid", error=error_msg, attempt=attempt)
                user_prompt = (
                    user_prompt + f"\n\nYour previous code failed with: {error_msg}\n"
                    "Please fix the code and output only valid Python."
                )
            else:
                logger.error("code_gen_ast_retry_failed", error=error_msg)

        raise PipelineError(
            "Generated code failed validation after 2 attempts: "
            f"{last_error or 'unknown validation error'}"
        )


def _validate_code_ast(code: str) -> str | None:
    """Validate generated Python code via AST + linting (no execution).

    Returns None if valid, or an error message string if not.
    Checks:
    - Valid Python syntax (AST parse)
    - Contains generate_features function
    - No banned imports (os, sys, etc.)
    - No undefined names (ruff lint, catches bare column refs like 'f1')
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Python syntax error: {exc}"

    # Check for generate_features function
    has_generate_fn = any(
        isinstance(node, ast.FunctionDef) and node.name == "generate_features"
        for node in ast.walk(tree)
    )
    if not has_generate_fn:
        return "Missing generate_features(df) function definition"

    # Check for banned imports
    _BANNED_IMPORTS = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "telnetlib",
        "pickle",
        "ctypes",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BANNED_IMPORTS:
                    return f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _BANNED_IMPORTS:
                    return f"Forbidden import: {node.module}"

    if shutil.which("ruff") is None:
        return None

    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "F821,F823", "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 and result.stdout.strip():
            first_error = (
                result.stdout.strip().split("\n")[0] if result.stdout.strip() else "lint error"
            )
            return f"Undefined name or reference: {first_error}"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass  # ruff not available, skip lint

    return None


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
        evaluator: CVEvaluator | None = None,
        sandbox: SandboxedExecutor | None = None,
        code_generator: CodeGenerator | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.evaluator = evaluator or CVEvaluator(config)
        self.sandbox = sandbox or SandboxedExecutor(
            timeout_seconds=config.evaluation.sandbox_timeout_seconds,
            max_memory_mb=config.evaluation.sandbox_max_memory_mb,
        )
        self.code_generator = code_generator or CodeGenerator(
            llm_client, max_tokens=config.llm.codegen_max_tokens
        )
        self._baseline_cache: dict[tuple[str, int, int], float] = {}

    def clear_baseline_cache(self) -> None:
        self._baseline_cache.clear()

    @staticmethod
    def _baseline_cache_key(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[str, int, int]:
        # Cache key includes a strong content fingerprint so in-place DataFrame mutations
        # across rounds cannot reuse stale baseline scores.
        x_hash = pd.util.hash_pandas_object(X_train, index=True, categorize=True).to_numpy()
        y_hash = pd.util.hash_pandas_object(y_train, index=True, categorize=True).to_numpy()
        digest = hashlib.blake2b(digest_size=16)
        digest.update(x_hash.tobytes())
        digest.update(y_hash.tobytes())
        digest.update(str(tuple(X_train.columns)).encode("utf-8"))
        digest.update(str(tuple(str(t) for t in X_train.dtypes)).encode("utf-8"))
        return (digest.hexdigest(), len(X_train), len(X_train.columns))

    async def run(
        self,
        agents: list[Agent],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame | None = None,
        context: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
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
        features_train, code, all_code_parts = await self._execute_train(
            all_specs, X_train, schema, semaphore
        )

        features_test = await self._execute_test(all_code_parts, X_test)

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
        )
        pipeline_latency_ms = round((time.perf_counter() - pipeline_t0) * 1000, 1)
        logger.info(
            "pipeline_complete",
            num_specs=len(all_specs),
            num_selected=len(result["selected_features_train"].columns)
            if not result["selected_features_train"].empty
            else 0,
            num_effective=len([g for g in result["gains"].values() if g > 0]),
            baseline_score=round(result["baseline_score"], 6),
            latency_ms=pipeline_latency_ms,
        )
        return result

    @staticmethod
    def _empty_result(X_train: pd.DataFrame, X_test: pd.DataFrame | None) -> dict[str, Any]:
        empty_train = pd.DataFrame(index=X_train.index)
        empty_test = pd.DataFrame(index=X_test.index if X_test is not None else X_train.index)
        return {
            "features_train": empty_train,
            "features_test": empty_test,
            "agent_gains": {},
            "specs": [],
            "top_features_train": empty_train,
            "top_features_test": empty_test,
            "baseline_score": 0.0,
            "gains": {},
            "generated_code": "",
        }

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
    ) -> tuple[pd.DataFrame, str, list[tuple[str, str]]]:
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
        all_code_parts: list[tuple[str, str]] = [r for r in code_gen_results if r is not None]

        sandbox_timeout = self.config.evaluation.sandbox_timeout_seconds

        exec_results = await asyncio.gather(
            *[
                _exec_sandbox(
                    self.sandbox, name, code, X_train, "malmas_core_train", sandbox_timeout
                )
                for name, code in all_code_parts
            ]
        )

        features_train_parts: list[pd.DataFrame] = []
        combined_code_parts: list[str] = []
        for (_name, code), result in zip(all_code_parts, exec_results, strict=True):
            if result is not None:
                features_train_parts.append(result[1])
                combined_code_parts.append(code)

        code = "\n\n".join(combined_code_parts)

        if not features_train_parts:
            raise PipelineError("All agent code executions failed — no features generated")

        features_train = self._concat_dedup(features_train_parts)

        logger.info(
            "code_generation_complete",
            num_specs=len(all_specs),
            num_agents=len(specs_by_agent),
            code_length=len(code),
            latency_ms=round((time.perf_counter() - code_gen_t0) * 1000, 1),
        )
        return features_train, code, all_code_parts

    async def _execute_test(
        self,
        all_code_parts: list[tuple[str, str]],
        X_test: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if X_test is None:
            return pd.DataFrame()

        sandbox_timeout = self.config.evaluation.sandbox_timeout_seconds
        test_exec_results = await asyncio.gather(
            *[
                _exec_sandbox(self.sandbox, name, code, X_test, "malmas_core_test", sandbox_timeout)
                for name, code in all_code_parts
            ],
            return_exceptions=True,
        )

        features_test_parts: list[pd.DataFrame] = []
        for test_result in test_exec_results:
            if isinstance(test_result, BaseException):
                logger.warning("agent_test_execution_task_failed", error=str(test_result)[:200])
                continue
            if test_result is not None:
                features_test_parts.append(test_result[1])
        features_test = self._concat_dedup(features_test_parts, index=X_test.index)
        return features_test

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
    ) -> dict[str, Any]:
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
        if (
            len(candidate_columns) > 1
            and self.config.evaluation.feature_eval_backend == "loky"
            and X_train.shape[0] * X_train.shape[1] > 2_000_000
        ):
            logger.warning(
                "feature_eval_backend_loky_large_matrix",
                rows=X_train.shape[0],
                cols=X_train.shape[1],
                candidates=len(candidate_columns),
                hint="Consider evaluation.feature_eval_backend='threading' to reduce IPC overhead",
            )
        eval_results = Parallel(n_jobs=n_jobs, backend=self.config.evaluation.feature_eval_backend)(
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
                gains[col] = float("-inf")
            else:
                gains[col] = result
                logger.debug(
                    "feature_evaluated",
                    feature=col,
                    gain=round(result, 6),
                    effective=result > 0,
                )

        effective = {k: v for k, v in gains.items() if v > 0}
        top_k = sorted(effective.items(), key=lambda x: x[1], reverse=True)
        top_k_names = [name for name, _ in top_k[: self.config.min_effective]]

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

        return {
            "features_train": features_train,
            "features_test": features_test,
            "all_features_train": features_train,
            "all_features_test": features_test if X_test is not None else pd.DataFrame(),
            "selected_features_train": top_features_train,
            "selected_features_test": top_features_test,
            "agent_gains": agent_gains,
            "specs": all_specs,
            "top_features_train": top_features_train,
            "top_features_test": top_features_test,
            "baseline_score": baseline_score,
            "gains": gains,
            "generated_code": code,
        }

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
        if features_train.empty:
            return []

        candidates = []
        for col in features_train.columns:
            series = features_train[col]
            if series.nunique(dropna=False) <= 1:
                continue
            candidates.append(col)

        max_candidates = self.config.evaluation.max_candidate_features
        if len(candidates) <= max_candidates:
            return candidates

        variances: list[tuple[str, float]] = []
        for col in candidates:
            series = features_train[col]
            if pd.api.types.is_numeric_dtype(series):
                variances.append((col, float(series.var(ddof=0))))
            else:
                variances.append((col, 0.0))
        variances.sort(key=lambda x: x[1], reverse=True)
        return [col for col, _ in variances[:max_candidates]]

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
