from typing import Literal

from pydantic import Field, model_validator

from feature_forge.methods._prompting import PromptParams, PromptRegistry, prompts_dir

_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry(prompts_dir(__package__))
    return _registry


class MalmusSingleShotParams(PromptParams):
    columns: str
    task: Literal["classification", "regression"]
    n_features: int = Field(default=5, ge=1)


class MalmusIterativeParams(PromptParams):
    columns: str
    task: Literal["classification", "regression"]
    n_iterations: int = Field(default=5, ge=1)
    iteration: int = Field(default=1, ge=1)
    existing_features: str = ""
    feedback: str = ""

    @model_validator(mode="after")
    def _validate_iteration_bounds(self) -> "MalmusIterativeParams":
        if self.iteration > self.n_iterations:
            raise ValueError("iteration must be <= n_iterations")
        return self


__all__ = [
    "MalmusIterativeParams",
    "MalmusSingleShotParams",
    "get_registry",
]
