from typing import Literal

from pydantic import BaseModel, Field, model_validator

from feature_forge.methods._prompting import PromptRegistry


def get_registry() -> PromptRegistry:
    """Cached prompt registry for the Malmus package."""
    return PromptRegistry.for_package(__package__)


class MalmusSingleShotParams(BaseModel):
    columns: str
    task: Literal["classification", "regression"]
    n_features: int = Field(default=5, ge=1)

    def render(self, template: str) -> str:
        return template.format(
            columns=self.columns,
            task=self.task,
            n_features=self.n_features,
        )


class MalmusIterativeParams(BaseModel):
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

    def render(self, template: str) -> str:
        return template.format(
            columns=self.columns,
            task=self.task,
            n_iterations=self.n_iterations,
            iteration=self.iteration,
            existing_features=self.existing_features,
            feedback=self.feedback,
        )
