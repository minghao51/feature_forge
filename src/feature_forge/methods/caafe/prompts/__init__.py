from pydantic import BaseModel, Field, model_validator

from feature_forge.methods._prompting import PromptRegistry


def get_registry() -> PromptRegistry:
    """Cached prompt registry for the CAAFE package."""
    return PromptRegistry.for_package(__package__)


class CAAFEUnifiedParams(BaseModel):
    description: str
    iterations: int = Field(default=2, ge=1)
    iteration: int = Field(default=1, ge=1)
    existing: str = ""
    feedback: str = ""

    @model_validator(mode="after")
    def _validate_iteration_bounds(self) -> "CAAFEUnifiedParams":
        if self.iteration > self.iterations:
            raise ValueError("iteration must be <= iterations")
        return self

    def render(self, template: str) -> str:
        return template.format(
            description=self.description,
            iterations=self.iterations,
            iteration=self.iteration,
            existing=self.existing,
            feedback=self.feedback,
        )
