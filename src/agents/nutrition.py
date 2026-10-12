from __future__ import annotations

from src.agents._common import invoke_structured
from src.agents.schemas import AnalysisOutput, UserPreferences


PROMPT = "Analyze dietary and nutrition fit. Never relax explicit dietary constraints."


def analyze_nutrition(candidates: list[dict], preferences: UserPreferences, llm) -> AnalysisOutput:
    return invoke_structured(
        llm, AnalysisOutput, PROMPT,
        {"candidates": candidates, "preferences": preferences.model_dump()},
    )
