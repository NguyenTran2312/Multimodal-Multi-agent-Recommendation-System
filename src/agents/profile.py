from __future__ import annotations

from src.agents._common import invoke_structured
from src.agents.schemas import ProfileOutput, UserPreferences


PROMPT = "Build a concise food-preference profile. Preserve hard dietary and budget constraints."


def build_profile(preferences: UserPreferences, llm) -> ProfileOutput:
    return invoke_structured(llm, ProfileOutput, PROMPT, preferences.model_dump())
