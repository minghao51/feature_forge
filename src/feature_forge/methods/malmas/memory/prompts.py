from feature_forge.methods._prompting import PromptParams

__all__ = [
    "SummarizeAgentParams",
    "SummarizeGlobalParams",
]


class SummarizeAgentParams(PromptParams):
    """Variables for ``malmas/prompts/summarize_agent.yaml``."""

    agent_name: str
    examples_text: str
    stats_text: str


class SummarizeGlobalParams(PromptParams):
    """Variables for ``malmas/prompts/summarize_global.yaml``."""

    combined_prompt: str
    task_description: str = ""
