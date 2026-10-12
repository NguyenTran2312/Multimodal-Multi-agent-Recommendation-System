from __future__ import annotations

from src.agents._common import invoke_structured
from src.agents.schemas import AnalysisOutput, UserPreferences


PROMPT = "Analyze which retrieved candidates best match current food-interest and popularity signals."


def analyze_trends(candidates: list[dict], preferences: UserPreferences, llm) -> AnalysisOutput:
    return invoke_structured(
        llm, AnalysisOutput, PROMPT,
        {"candidates": candidates, "preferences": preferences.model_dump()},
    )
