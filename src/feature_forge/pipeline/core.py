"""Core pipeline for single-round feature engineering.

Orchestrates agents, code generation, sandboxed execution,
and feature evaluation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from feature_forge.agents.base import Agent
from feature_forge.config import Settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.sandbox import SandboxedExecutor
from feature_forge.exceptions import PipelineError
from feature_forge.llm.base import LLMClient
from feature_forge.types import FeatureSpec


class CodeGenerator:
    """Generates pandas code from feature specifications."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        prompt_path = Path(__file__).parent / "../prompts/code_generation.txt"
        self._system_prompt = (
            prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
        )

    async def generate_code(self, specs: list[FeatureSpec]) -> str:
        """Generate Python code for a list of feature specs."""
        user_prompt = f"Please generate code for the following features:\n{specs}"
        response = await self.llm_client.complete(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        return response.content


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
        self.sandbox = sandbox or SandboxedExecutor()
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
            if isinstance(specs, Exception):
                continue
            for spec in specs:
                spec["agent_name"] = agent.name
            all_specs.extend(specs)

        if not all_specs:
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

        # Step 2: Generate code for all specs
        try:
            code = await self.code_generator.generate_code(all_specs)
        except Exception as exc:
            raise PipelineError(f"Code generation failed: {exc}") from exc

        # Step 3: Execute code to get feature DataFrames
        try:
            features_train = self.sandbox.execute(code, X_train)
        except Exception as exc:
            raise PipelineError(f"Sandbox execution failed: {exc}") from exc

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
        gains: dict[str, float] = {}
        for col in features_train.columns:
            try:
                gain = self.evaluator.evaluate_feature(
                    X_train, y_train, features_train[[col]], baseline_score=baseline_score
                )
                gains[col] = gain
            except Exception:
                gains[col] = float("-inf")

        # Step 5: Select top-k effective features
        effective = {k: v for k, v in gains.items() if v > 0}
        top_k = sorted(effective.items(), key=lambda x: x[1], reverse=True)
        top_k_names = [name for name, _ in top_k[: self.config.min_effective]]

        top_features_train = features_train[top_k_names] if top_k_names else pd.DataFrame(index=X_train.index)
        top_features_test = features_test[top_k_names] if top_k_names and X_test is not None else pd.DataFrame()

        # Build per-agent gain DataFrames
        agent_gains: dict[str, pd.DataFrame] = {}
        for agent in agents:
            agent_feature_names = [s["name"] for s in all_specs if s.get("agent_name") == agent.name]
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
            "agent_gains": agent_gains,
            "specs": all_specs,
            "top_features_train": top_features_train,
            "top_features_test": top_features_test,
            "baseline_score": baseline_score,
            "gains": gains,
            "generated_code": code,
        }
