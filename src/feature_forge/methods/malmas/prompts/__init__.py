from feature_forge.methods._prompting import PromptRegistry


def get_registry() -> PromptRegistry:
    """Cached prompt registry for the MALMAS package."""
    return PromptRegistry.for_package(__package__)
