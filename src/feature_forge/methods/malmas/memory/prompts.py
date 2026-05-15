from pydantic import BaseModel


class SummarizeAgentParams(BaseModel):
    agent_name: str
    examples_text: str
    stats_text: str

    def render_system(self) -> str:
        return (
            f"You are {self.agent_name} agent, an expert feature engineering assistant. "
            "You will receive a list of effective features and statistics about their patterns. "
            "Your task is to generate effective, high-quality conceptual rules using concise language "
            "that can guide future feature generation. Rules should directly reflect the statistics and examples. "
            "Avoid any irrelevant information."
        )

    def render_user(self) -> str:
        return (
            f"Here are the effective feature examples:\n\n{self.examples_text}\n\n"
            f"Here are the statistics about effective features:\n\n{self.stats_text}\n\n"
            "Based on both the examples and the statistics, summarize 1 to 3 concise and actionable "
            "conceptual rules to optimize future feature generation. Rules should be in clear bullet points."
        )


class SummarizeGlobalParams(BaseModel):
    combined_prompt: str
    task_description: str = ""

    def render_system(self) -> str:
        return (
            "You are a senior AutoML optimization assistant. "
            "You will receive conceptual summaries and statistics from multiple feature engineering agents. "
            "Your task is to synthesize these into 2 to 5 concise, effective, high-level conceptual rules "
            "that can guide future global feature derivation tasks across all agents. Avoid any irrelevant information."
        )

    def render_user(self) -> str:
        return (
            f"The description of this dataset is:\n{self.task_description}\n"
            f"Here are the conceptual summaries and statistics from all agents:\n\n{self.combined_prompt}\n\n"
            "Based on the above, summarize 2 to 5 concise, actionable, high-level conceptual rules "
            "for optimizing future feature generation across all agents. Rules should be in clear bullet points."
        )
