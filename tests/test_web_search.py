from src.tools.web_search import TavilyWebSearchClient


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
    assert rows[0].title == "A"
    assert rows[0].source == "tavily"
    assert rows[0].published_at == "2026-01-01"


def test_tavily_requires_key_only_when_constructed():
    try:
        TavilyWebSearchClient("")
    except ValueError as exc:
        assert "TAVILY_API_KEY" in str(exc)
    else:
        raise AssertionError("expected ValueError")
