from pydantic import BaseModel

from src.tools.web_search import (
    RestaurantWebEvidence,
    TavilyWebSearchClient,
    WebSearchResult,
)


def test_tavily_normalizes_and_caps_results():
    def post(url, *, json, timeout):
        assert json["query"] == "pho"
        return {
            "results": [
                {"title": "A", "url": "https://a", "content": "alpha", "published_date": "2026-01-01"},
                {"title": "B", "url": "https://b", "content": "beta"},
                {"title": "", "url": "", "content": ""},
            ]
        }

    rows = TavilyWebSearchClient("key", post=post).search("pho", max_results=1)
    assert len(rows) == 1
    assert isinstance(rows[0], BaseModel)
    assert isinstance(rows[0], WebSearchResult)
    assert rows[0].title == "A"
    assert rows[0].source == "tavily"
    assert rows[0].published_at == "2026-01-01"


def test_restaurant_web_evidence_is_strictly_grounded():
    result = WebSearchResult(
        title="Pho guide",
        url="https://guide",
        snippet="A page discussing Pho House in District 1.",
        source="tavily",
    )
    evidence = RestaurantWebEvidence.from_search_result(result)
    assert evidence.name is None
    assert evidence.location is None
    assert evidence.cuisine is None
    assert evidence.price is None
    assert evidence.rating is None
    assert evidence.opening_hours is None
    assert evidence.menu_items == []
    assert evidence.description == result.snippet
    assert evidence.source_url == result.url
    assert evidence.source == "tavily"


def test_tavily_requires_key_only_when_constructed():
    try:
        TavilyWebSearchClient("")
    except ValueError as exc:
        assert "TAVILY_API_KEY" in str(exc)
    else:
        raise AssertionError("expected ValueError")
