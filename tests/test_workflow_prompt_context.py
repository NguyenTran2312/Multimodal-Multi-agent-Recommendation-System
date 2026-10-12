from src.agents.schemas import Recommendation, RecommendationResponse, UserPreferences
from src.workflow.recommendation import RecommendationWorkflow


def test_workflow_passes_original_query_to_final_recommender():
    seen = {}

    def recommender(candidates, analyses, preferences):
        seen["query"] = analyses.get("user_query")
        return RecommendationResponse(
            recommendations=[Recommendation(candidate_id="r1", reason="fit", rank=1)]
        )

    workflow = RecommendationWorkflow(
        preference_extractor=lambda query: UserPreferences(cuisine=["Vietnamese"]),
        retrieve=lambda query, preferences: [{"id": "r1", "record": {"name": "Pho House"}}],
        profile_agent=lambda preferences: object(),
        trend_agent=lambda candidates, preferences: object(),
        style_agent=lambda candidates, preferences: object(),
        nutrition_agent=lambda candidates, preferences: object(),
        recommender=recommender,
    )

    workflow.recommend("Tôi muốn món Việt Nam")
    assert seen["query"] == "Tôi muốn món Việt Nam"
