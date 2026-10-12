from __future__ import annotations

from src.agents._common import invoke_structured
from src.agents.prompt_policy import compose_prompt
from src.agents.schemas import AnalysisOutput, UserPreferences


PROMPT = compose_prompt(
    """
TASK: Analyze dietary and nutrition fit using only known ingredients, descriptions, and
structured metadata from the retrieved candidates.

Never relax an explicit dietary constraint. High-level observations such as vegetable-rich,
protein-containing, creamy, fried, or spicy are acceptable only when directly supported by
the input. Do not invent calorie counts, macro or macronutrient values, an allergen guarantee,
a medical benefit, a health outcome, or disease-related advice. If the available evidence is
insufficient to assess a nutrition or allergen question, state that limitation briefly instead
of guessing.
"""
)


def analyze_nutrition(candidates: list[dict], preferences: UserPreferences, llm) -> AnalysisOutput:
    return invoke_structured(
        llm, AnalysisOutput, PROMPT,
        {"candidates": candidates, "preferences": preferences.model_dump()},
    )
