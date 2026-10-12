from types import SimpleNamespace

from src.agents.schemas import UserPreferences
from src.retrieval.external import (
    deduplicate_candidates,
    place_to_candidate,
    should_use_external_search,
    web_result_to_evidence,
)
from src.tools.places import PlaceResult
from src.tools.web_search import WebSearchResult
from src.ui.chatbot import retrieve_candidates


def test_external_routing_prefers_local_exact_unless_freshness_requested():
    exact = [{"id": "r1", "match_type": "exact"}]
    assert should_use_external_search("Italian dinner", exact, SimpleNamespace()) is False
    assert should_use_external_search("open now Italian dinner", exact, SimpleNamespace()) is True
    assert should_use_external_search("quán thật gần đây", exact, SimpleNamespace()) is True
    assert should_use_external_search("Italian dinner", [], SimpleNamespace()) is True
    assert should_use_external_search(
        "Italian dinner", [{"id": "x", "match_type": "relaxed"}], SimpleNamespace()
    ) is True


def test_external_normalization_and_deduplication():
    place = PlaceResult(
        "p1", "Pho House", "1 Main", ["restaurant"], None, 4.5, 10,
        "https://p", None, []
    )
    candidate = place_to_candidate(place)
    assert candidate["id"] == "google_places:p1"
    assert candidate["record"]["cuisine"] is None
    assert candidate["record"]["source"] == "google_places"

    evidence = web_result_to_evidence(
        WebSearchResult("Guide", "https://g", "text", "tavily")
    )
    assert evidence["source_url"] == "https://g"
    assert len(deduplicate_candidates([candidate, candidate])) == 1


def test_retrieve_candidates_uses_external_clients_when_local_results_are_empty():
    class Store:
        def has_collection(self, name):
            return False

        def search(self, kind, query, preferences, **kwargs):
            return []

    class WebClient:
        def search(self, query, *, max_results):
            return [WebSearchResult("Guide", "https://g", "fresh evidence", "tavily")]

    class PlacesClient:
        def search(self, query, *, max_results):
            return [
                PlaceResult(
                    "p1", "Pho House", "1 Main", ["restaurant"], None,
                    4.5, 10, "https://p", None, []
                )
            ]

    results = retrieve_candidates(
        Store(),
        "real Vietnamese restaurant nearby",
        UserPreferences(),
        web_search_client=WebClient(),
        places_client=PlacesClient(),
    )
    assert results[0]["id"] == "google_places:p1"
    assert results[0]["external_evidence"][0]["source_url"] == "https://g"
