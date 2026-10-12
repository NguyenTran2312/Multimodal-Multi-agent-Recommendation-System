from types import SimpleNamespace

from src.retrieval.external import (
    deduplicate_candidates,
    place_to_candidate,
    should_use_external_search,
    web_result_to_evidence,
)
from src.tools.places import PlaceResult
from src.tools.web_search import WebSearchResult


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
