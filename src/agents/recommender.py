from __future__ import annotations

from src.agents._common import invoke_structured
from src.agents.prompt_policy import compose_prompt
from src.agents.schemas import RecommendationResponse, UserPreferences


PROMPT = compose_prompt(
    """
TASK: Produce the final user-facing culinary recommendations from only the provided
candidate IDs, analyses, preferences, and original user query when present.

Recommend the strongest relevant options only; do not add a weak alternative merely to
fill the list. Keep the response to the supplied candidate set and at most the configured
number of results. For each selected candidate, write one concise reason explaining the fit
using supported user-facing details such as cuisine, ingredients, price, location,
atmosphere, preparation time, dietary fit, or review evidence.

Use a professional but approachable culinary-consultant voice. Match the same language as
the original user query when it is provided; for mixed-language requests, answer in the
dominant user language while preserving familiar food and cuisine names.

Do not mention ranking score, fusion score, modality score, semantic score, embedding or
embeddings, candidate ID or candidate IDs, agents, vector search, retrieval internals, or
implementation details. Do not claim that something is popular, trending, viral, or in
high demand unless the supplied records contain explicit evidence for that claim.
"""
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
