from __future__ import annotations

from src.agents._common import invoke_structured
from src.agents.schemas import AnalysisOutput, UserPreferences


PROMPT = "Analyze atmosphere, cuisine style, and visual fit for the retrieved candidates."


def analyze_style(candidates: list[dict], preferences: UserPreferences, llm) -> AnalysisOutput:
    return invoke_structured(
        llm, AnalysisOutput, PROMPT,
        {"candidates": candidates, "preferences": preferences.model_dump()},
    )
