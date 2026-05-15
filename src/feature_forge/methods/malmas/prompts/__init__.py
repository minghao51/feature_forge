from feature_forge.methods._prompting import PromptRegistry, prompts_dir

_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry(prompts_dir(__package__))
    return _registry
