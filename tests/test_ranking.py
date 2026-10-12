from src.retrieval.ranking import preference_score, rerank_candidates


def test_soft_budget_mismatch_keeps_semantically_strong_candidate():
    candidates = [
        {"id": "cheap-spicy", "fusion_score": 1.0, "record": {"price": "$", "description": "spicy Sichuan noodles"}},
        {"id": "moderate", "fusion_score": 0.95, "record": {"price": "$$", "description": "spicy noodles"}},
    ]
    ranked = rerank_candidates(candidates, {"budget": "moderate"}, dish_terms=["spicy", "noodles"])
    assert {x["id"] for x in ranked} == {"cheap-spicy", "moderate"}
    assert ranked[0]["id"] == "moderate"
    assert next(x for x in ranked if x["id"] == "cheap-spicy")["match_type"] == "relaxed"


def test_preference_score_missing_fields_is_safe():
    assert preference_score({}, {"budget": "cheap", "location": "HCMC"}) == 0.0
