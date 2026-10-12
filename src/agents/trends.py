from __future__ import annotations

from src.agents._common import invoke_structured
from src.agents.prompt_policy import compose_prompt
from src.agents.schemas import AnalysisOutput, UserPreferences


PROMPT = compose_prompt(
    """
TASK: Analyze the strength of each retrieved candidate using only evidence present in the
provided candidate records and user preferences.

Use supplied ratings, reviews, descriptions, and other explicit evidence when relevant.
Treat a rating as an observed rating only; it is not proof that something is currently
popular. Do not claim that a candidate is trending, viral, popular, fashionable, or in
high demand unless the input explicitly provides evidence for that claim. Do not use or
explain internal ranking, retrieval, or model scores. Summarize only evidence that can help
the final recommender judge user fit.
"""
)


def analyze_trends(candidates: list[dict], preferences: UserPreferences, llm) -> AnalysisOutput:
    return invoke_structured(
        llm, AnalysisOutput, PROMPT,
        {"candidates": candidates, "preferences": preferences.model_dump()},
    )
