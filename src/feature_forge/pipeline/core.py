"""Core pipeline for single-round feature engineering.

Orchestrates agents, code generation, sandboxed execution,
and feature evaluation.
"""

from __future__ import annotations

import ast
import asyncio
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed  # type: ignore[import-untyped]

from feature_forge.agents.base import Agent
from feature_forge.config import Settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import PipelineError
from feature_forge.llm.base import LLMClient
from feature_forge.observability.structlog_config import get_logger
from feature_forge.types import FeatureSpec
from feature_forge.utils import strip_markdown_fences

logger = get_logger(__name__)


class CodeGenerator:
    """Generates pandas code from feature specifications."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        prompt_path = Path(__file__).parent / "../prompts/code_generation.txt"
        self._system_prompt = (
            prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
        )

    async def generate_code(
        self, specs: list[FeatureSpec], error_feedback: str | None = None
    ) -> str:
        """Generate Python code for a list of feature specs.

        Includes AST validation and a single retry on syntax errors.
        """
        specs_dump = [s.model_dump() if hasattr(s, "model_dump") else s for s in specs]
        user_prompt = f"Please generate code for the following features:\n{specs_dump}"
        if error_feedback:
            user_prompt += f"\n\n{error_feedback}"

        for attempt in range(2):
            response = await self.llm_client.complete(
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            code = strip_markdown_fences(response.content)

            # Validate with AST
            error_msg = _validate_code_ast(code)
            if error_msg is None:
                return code

            if attempt == 0:
                logger.warning("code_gen_ast_invalid", error=error_msg, attempt=attempt)
                user_prompt = (
                    user_prompt + f"\n\nYour previous code failed with: {error_msg}\n"
                    "Please fix the code and output only valid Python."
                )
            else:
                logger.error("code_gen_ast_retry_failed", error=error_msg)

        return code


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

    # Ruff lint: catches undefined names (bare 'f1' instead of df['f1'])
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "F821,F823", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 and result.stdout.strip():
            # Extract first error line
            first_error = (
                result.stdout.strip().split("\n")[0] if result.stdout.strip() else "lint error"
            )
            return f"Undefined name or reference: {first_error}"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass  # ruff not available, skip lint
    finally:
        try:
            import os as _os

            _os.unlink(tmp_path)
        except (OSError, NameError):
            pass

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
        self.code_generator = code_generator or CodeGenerator(llm_client)

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
        if isinstance(context, list):
            per_agent_contexts = context
        else:
            per_agent_contexts = [context or {}] * len(agents)

        semaphore = asyncio.Semaphore(self.config.llm.max_concurrent_calls)

        # Step 1: Generate feature specs from all agents in parallel
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

        if not all_specs:
            logger.info(
                "pipeline_complete", num_specs=0, num_selected=0, reason="no_specs_generated"
            )
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

        # Step 2-3: Generate code + execute in sandbox (with retry on failure)
        logger.info("code_generation_start", num_specs=len(all_specs))
        code_gen_t0 = time.perf_counter()
        code = ""
        features_train = pd.DataFrame()
        last_error: str | None = None

        for code_attempt in range(3):
            error_feedback = (
                f"Previous attempt failed with: {last_error}. Fix the error and regenerate."
                if last_error
                else None
            )
            try:
                code = await self.code_generator.generate_code(
                    all_specs, error_feedback=error_feedback
                )
            except Exception as exc:
                raise PipelineError(f"Code generation failed: {exc}") from exc

            sandbox_t0 = time.perf_counter()
            try:
                features_train = self.sandbox.execute(code, X_train)
                break  # success
            except Exception as exc:
                last_error = str(exc)
                if code_attempt < 2:
                    logger.warning(
                        "sandbox_execution_retry",
                        attempt=code_attempt,
                        error=last_error[:200],
                    )
                else:
                    raise PipelineError(
                        f"Sandbox execution failed after 3 attempts: {last_error}"
                    ) from exc

        if features_train.empty:
            raise PipelineError(f"Sandbox execution failed after 3 attempts: {last_error}")

        logger.info(
            "code_generation_complete",
            num_specs=len(all_specs),
            code_length=len(code),
            latency_ms=round((time.perf_counter() - code_gen_t0) * 1000, 1),
        )
        logger.info(
            "sandbox_execution_complete",
            result_shape=features_train.shape,
            latency_ms=round((time.perf_counter() - sandbox_t0) * 1000, 1),
        )

        features_test = pd.DataFrame()
        if X_test is not None:
            try:
                features_test = self.sandbox.execute(code, X_test)
            except Exception:
                features_test = pd.DataFrame(index=X_test.index)

        # Ensure alignment
        features_train = features_train.reindex(X_train.index)
        if X_test is not None:
            features_test = features_test.reindex(X_test.index)

        # Step 4: Evaluate each feature
        baseline_score = self.evaluator.evaluate_baseline(X_train, y_train)
        logger.info(
            "evaluation_baseline", score=round(baseline_score, 6), metric=self.config.metric
        )
        gains: dict[str, float] = {}
        candidate_columns = self._prefilter_candidate_columns(features_train)

        if len(candidate_columns) > 1:
            eval_results = Parallel(n_jobs=-1, prefer="threads")(
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
        else:
            for col in candidate_columns:
                try:
                    gain = self.evaluator.evaluate_feature(
                        X_train, y_train, features_train[[col]], baseline_score=baseline_score
                    )
                    gains[col] = gain
                    logger.debug(
                        "feature_evaluated", feature=col, gain=round(gain, 6), effective=gain > 0
                    )
                except Exception as exc:
                    logger.warning("feature_evaluation_failed", feature=col, error=str(exc))
                    if self.config.evaluation.fail_on_feature_error:
                        raise PipelineError(
                            f"Feature evaluation failed for '{col}': {exc}"
                        ) from exc
                    gains[col] = float("-inf")

        # Step 5: Select top-k effective features
        effective = {k: v for k, v in gains.items() if v > 0}
        top_k = sorted(effective.items(), key=lambda x: x[1], reverse=True)
        top_k_names = [name for name, _ in top_k[: self.config.min_effective]]

        top_features_train = (
            features_train[top_k_names] if top_k_names else pd.DataFrame(index=X_train.index)
        )
        top_features_test = (
            features_test[top_k_names] if top_k_names and X_test is not None else pd.DataFrame()
        )

        # Build per-agent gain DataFrames
        agent_gains: dict[str, pd.DataFrame] = {}
        for agent in agents:
            agent_feature_names = [s.name for s in all_specs if s.agent_name == agent.name]
            agent_gain_rows = []
            for fname in agent_feature_names:
                if fname in gains:
                    agent_gain_rows.append({"feature": fname, "gain": gains[fname]})
            if agent_gain_rows:
                agent_gains[agent.name] = pd.DataFrame(agent_gain_rows)

        num_effective = len([g for g in gains.values() if g > 0])
        pipeline_latency_ms = round((time.perf_counter() - pipeline_t0) * 1000, 1)
        logger.info(
            "pipeline_complete",
            num_specs=len(all_specs),
            num_selected=len(top_k_names),
            num_effective=num_effective,
            baseline_score=round(baseline_score, 6),
            latency_ms=pipeline_latency_ms,
        )

        return {
            "features_train": features_train,
            "features_test": features_test,
            "all_features_train": features_train,
            "all_features_test": features_test if X_test is not None else pd.DataFrame(),
            "agent_gains": agent_gains,
            "specs": all_specs,
            "top_features_train": top_features_train,
            "top_features_test": top_features_test,
            "baseline_score": baseline_score,
            "gains": gains,
            "generated_code": code,
        }

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
