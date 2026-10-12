from __future__ import annotations

from src.agents._common import invoke_structured
from src.agents.schemas import RecommendationResponse, UserPreferences


PROMPT = (
    "Rank only the provided candidate IDs. Explain the recommendation from the "
    "retrieved records and analyses; do not invent candidates."
)


def generate_recommendations(
    candidates: list[dict],
    analyses: dict,
    preferences: UserPreferences,
    llm,
) -> RecommendationResponse:
    result = invoke_structured(
        llm,
        RecommendationResponse,
        PROMPT,
        {
            "candidates": candidates,
            "analyses": analyses,
            "preferences": preferences.model_dump(),
        },
    )
    allowed = {str(candidate["id"]) for candidate in candidates}
    for recommendation in result.recommendations:
        if recommendation.candidate_id not in allowed:
            raise ValueError(
                f"Recommendation candidate {recommendation.candidate_id!r} is not in retrieved candidates"
            )
    return result
