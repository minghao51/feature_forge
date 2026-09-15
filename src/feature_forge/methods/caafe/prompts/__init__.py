from pydantic import Field, model_validator

from feature_forge.methods._prompting import PromptParams, PromptRegistry, prompts_dir

_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry(prompts_dir(__package__))
    return _registry


class CAAFEUnifiedParams(PromptParams):
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


__all__ = [
    "CAAFEUnifiedParams",
    "get_registry",
]
