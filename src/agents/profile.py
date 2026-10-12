from __future__ import annotations

from src.agents._common import invoke_structured
from src.agents.prompt_policy import compose_prompt
from src.agents.schemas import ProfileOutput, UserPreferences


PROMPT = compose_prompt(
    """
TASK: Build a concise food-preference profile for downstream recommendation reasoning.

Prioritize explicit hard constraints first, then summarize soft preferences separately.
Use only information present in the structured preferences. Do not infer or invent personal
attributes such as age, lifestyle, health goals, spending habits, family status, or dining
habits that the user did not state. Preserve dietary and budget constraints exactly as
provided. Keep the summary short and action-oriented.
"""
)


def build_profile(preferences: UserPreferences, llm) -> ProfileOutput:
    return invoke_structured(llm, ProfileOutput, PROMPT, preferences.model_dump())
