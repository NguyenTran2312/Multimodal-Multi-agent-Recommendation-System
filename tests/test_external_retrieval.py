from types import SimpleNamespace

from src.retrieval.external import should_use_external_search, web_result_to_evidence
from src.tools.web_search import RestaurantWebEvidence, WebSearchResult


def test_external_routing_prefers_local_exact_unless_freshness_requested():
    exact = [{"id": "r1", "match_type": "exact"}]
    assert should_use_external_search("Italian dinner", exact, SimpleNamespace()) is False
    assert should_use_external_search("open now Italian dinner", exact, SimpleNamespace()) is True
    assert should_use_external_search("quán thật gần đây", exact, SimpleNamespace()) is True
    assert should_use_external_search("Italian dinner", [], SimpleNamespace()) is True
    assert should_use_external_search(
        "Italian dinner", [{"id": "x", "match_type": "relaxed"}], SimpleNamespace()
    ) is True


def test_web_result_becomes_structured_restaurant_evidence_without_invention():
    evidence = web_result_to_evidence(
        WebSearchResult(
            title="Guide",
            url="https://g",
            snippet="A current restaurant guide.",
        )
    )
    assert isinstance(evidence, RestaurantWebEvidence)
    assert evidence.source_url == "https://g"
    assert evidence.description == "A current restaurant guide."
    assert evidence.name is None
    assert evidence.location is None
    assert evidence.rating is None
    assert evidence.menu_items == []
