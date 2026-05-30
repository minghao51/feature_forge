from __future__ import annotations

from dataclasses import dataclass

from feature_forge.config import Settings
from feature_forge.evaluation.cv import CVEvaluator
from feature_forge.evaluation.model_factory import ModelFactory
from feature_forge.evaluation.sandbox import SandboxedExecutor


@dataclass
class EvaluationKit:
    sandbox: SandboxedExecutor
    evaluator: CVEvaluator
    model_factory: ModelFactory

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> EvaluationKit:
        settings = settings or Settings()
        model_factory = ModelFactory(random_state=settings.random_state)
        evaluator = CVEvaluator(config=settings, model_factory=model_factory)
        sandbox = SandboxedExecutor(
            timeout_seconds=settings.evaluation.sandbox_timeout_seconds,
            max_memory_mb=settings.evaluation.sandbox_max_memory_mb,
        )
        return cls(sandbox=sandbox, evaluator=evaluator, model_factory=model_factory)
