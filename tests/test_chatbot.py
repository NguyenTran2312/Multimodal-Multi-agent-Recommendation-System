import pytest

from src.agents.schemas import Recommendation, RecommendationResponse, UserPreferences
from src.tools.web_search import RestaurantWebEvidence, WebSearchResult
from src.ui.chatbot import (
    build_runtime,
    classify_intent,
    format_workflow_result,
    handle_message,
    retrieve_candidates,
)
from src.workflow.recommendation import WorkflowError


def test_format_success_uses_only_returned_candidates():
    result = {
        "status": "ok",
        "candidates": [
            {"id": "r1", "record": {"name": "Sakura", "cuisine": "Japanese"}},
            {"id": "r2", "record": {"name": "Other"}},
        ],
        "recommendations": RecommendationResponse(
            recommendations=[Recommendation(candidate_id="r1", reason="Cozy and affordable", rank=1)]
        ),
    }
    text = format_workflow_result(result)
    assert "Sakura" in text
    assert "Cozy and affordable" in text
    assert "Other" not in text


def test_format_no_match_message():
    assert "couldn't find" in format_workflow_result({"status": "no_match", "candidates": []}).lower()


def test_build_runtime_missing_key_is_actionable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_runtime()


def test_handle_message_turns_workflow_failure_into_user_message():
    class BrokenWorkflow:
        def recommend(self, _query):
            raise WorkflowError("provider unavailable")

    text = handle_message("ramen", BrokenWorkflow())
    assert "provider unavailable" in text
    assert "could not complete" in text.lower()


def test_format_rejects_recommendation_missing_from_candidates():
    result = {
        "status": "ok",
        "candidates": [{"id": "r1", "record": {"name": "Sakura"}}],
        "recommendations": RecommendationResponse(
            recommendations=[Recommendation(candidate_id="x", reason="invented", rank=1)]
        ),
    }
    with pytest.raises(ValueError, match="not present"):
        format_workflow_result(result)


def test_smoke_script_runs_directly_from_repo_root_without_pythonpath(monkeypatch):
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "scripts/smoke_test.py", "ramen"],
        cwd=".",
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "OPENAI_API_KEY" in result.stdout


def test_format_relaxed_candidate_explains_mismatch():
    result = {
        "status": "ok",
        "candidates": [{"id": "r1", "record": {"name": "Noodle Bar"}, "match_type": "relaxed", "preference_mismatches": ["budget"]}],
        "recommendations": RecommendationResponse(recommendations=[Recommendation(candidate_id="r1", reason="Spicy noodles", rank=1)]),
    }
    text = format_workflow_result(result)
    assert "closest match" in text.lower() and "budget" in text.lower()


def test_format_genuine_no_match_mentions_indexed_data():
    text = format_workflow_result({"status": "no_match", "candidates": []})
    assert "indexed data" in text.lower()


def test_classify_intent_distinguishes_restaurants_dishes_and_recipes():
    assert classify_intent("Find me a Vietnamese restaurant for dinner") == "restaurant"
    assert classify_intent("What Vietnamese dish has beef noodles and spice?") == "dish"
    assert classify_intent("I want a vegetarian Vietnamese meal") == "dish"
    assert classify_intent("Something similar to bún bò Huế but less spicy") == "dish"
    assert classify_intent("Give me a recipe I can cook at home") == "recipe"


def test_retrieve_candidates_returns_dishes_for_dish_intent():
    class Store:
        def has_collection(self, name): return name in {"dish_text", "dish_image"}
        def search(self, kind, query, preferences, **kwargs):
            if kind != "dish" or kwargs.get("modality") != "text":
                return []
            return [{
                "id": "d1",
                "record": {"name": "Bún bò Huế", "name_vi": "Bún bò Huế"},
                "document": "Bún bò Huế spicy beef noodle soup",
                "distance": 0.1,
                "modality": "text",
            }]

    results = retrieve_candidates(
        Store(),
        "What Vietnamese dish would you recommend if I want noodles, beef, and something spicy?",
        UserPreferences(cuisine=["Vietnamese"]),
    )
    assert results and results[0]["id"] == "d1"


def test_retrieve_candidates_uses_dish_knowledge_and_keeps_soft_budget_out_of_where():
    class Store:
        def __init__(self): self.calls = []
        def has_collection(self, name): return name == "dish_text"
        def search(self, kind, query, preferences, **kwargs):
            self.calls.append((kind, preferences, kwargs))
            if kind == "dish":
                return [{"id": "d1", "record": {"name": "Bún bò Huế"}, "document": "Bún bò Huế", "distance": 0.1, "modality": "text"}]
            if kwargs.get("modality") == "text":
                return [{"id": "r1", "record": {"name": "Noodle Bar", "price": "$", "description": "spicy Bún bò Huế noodles"}, "document": "spicy Bún bò Huế noodles", "distance": 0.1, "modality": "text"}]
            return []

    store = Store()
    result = retrieve_candidates(store, "cheap bún bò Huế restaurant", UserPreferences(budget="cheap"))
    assert result and result[0]["id"] == "r1"
    restaurant_calls = [call for call in store.calls if call[0] == "restaurants"]
    assert all(call[1] == {} for call in restaurant_calls)
    assert any(call[0] == "dish" for call in store.calls)


def test_retrieve_candidates_without_dish_collection_still_works():
    class Store:
        def has_collection(self, name): return False
        def search(self, kind, query, preferences, **kwargs):
            if kwargs.get("modality") == "text":
                return [{"id": "r1", "record": {"name": "Cafe", "price": "$$"}, "document": "casual cafe", "distance": 0.2, "modality": "text"}]
            return []
    assert retrieve_candidates(Store(), "casual restaurant", UserPreferences())[0]["id"] == "r1"


def test_retrieve_candidates_applies_explicit_strict_constraints():
    class Store:
        def has_collection(self, name): return False
        def search(self, kind, query, preferences, **kwargs):
            assert kwargs["strict_only"] is True
            assert preferences == {"strict_constraints": {"location": "District 1"}}
            return []
    prefs = UserPreferences(location="District 1", strict_constraints={"location": "District 1"})
    assert retrieve_candidates(Store(), "must be in District 1", prefs) == []


def test_retrieve_candidates_returns_only_exact_matches_when_available_and_caps_at_three():
    class Store:
        def has_collection(self, name): return False
        def search(self, kind, query, preferences, **kwargs):
            if kwargs.get("modality") != "text":
                return []
            return [
                {"id": "r1", "record": {"name": "Exact 1", "cuisine": "Italian"}, "document": "Italian", "distance": 0.1, "modality": "text"},
                {"id": "r2", "record": {"name": "Exact 2", "cuisine": "Italian"}, "document": "Italian", "distance": 0.2, "modality": "text"},
                {"id": "r3", "record": {"name": "Fallback 1", "cuisine": "Japanese"}, "document": "Japanese", "distance": 0.3, "modality": "text"},
                {"id": "r4", "record": {"name": "Fallback 2", "cuisine": "American"}, "document": "American", "distance": 0.4, "modality": "text"},
            ]

    results = retrieve_candidates(Store(), "Italian restaurant", UserPreferences(cuisine=["Italian"]), retrieval_k=3)
    assert [item["id"] for item in results] == ["r1", "r2"]


def test_tavily_runs_when_local_restaurant_results_are_empty_and_creates_grounded_candidates():
    class Store:
        def has_collection(self, name): return False
        def search(self, *args, **kwargs): return []

    class WebClient:
        def search(self, query, *, max_results):
            assert "District 1" in query
            return [WebSearchResult(title="Pho Viet", url="https://example.com/pho", snippet="Vietnamese restaurant in District 1")]

    def extract(results, query):
        assert results[0].url == "https://example.com/pho"
        return [RestaurantWebEvidence(
            name="Pho Viet",
            location="District 1",
            cuisine="Vietnamese",
            description="Vietnamese restaurant in District 1",
            source_url="https://example.com/pho",
        )]

    results = retrieve_candidates(
        Store(),
        "Find a real Vietnamese restaurant that is open now in District 1",
        UserPreferences(cuisine=["Vietnamese"], location="District 1"),
        web_search_client=WebClient(),
        web_candidate_extractor=extract,
    )
    assert results
    assert results[0]["record"]["name"] == "Pho Viet"
    assert results[0]["record"]["source_url"] == "https://example.com/pho"


def test_explicit_restaurant_cuisine_mismatch_is_not_returned_as_weak_fallback():
    class Store:
        def has_collection(self, name): return False
        def search(self, kind, query, preferences, **kwargs):
            if kwargs.get("modality") == "text":
                return [{
                    "id": "sichuan",
                    "record": {"name": "Neon Noodle Bar", "cuisine": "Sichuan"},
                    "document": "spicy noodles",
                    "distance": 0.1,
                    "modality": "text",
                }]
            return []

    results = retrieve_candidates(
        Store(),
        "Find me a Vietnamese restaurant with spicy food",
        UserPreferences(cuisine=["Vietnamese"]),
    )
    assert results == []
