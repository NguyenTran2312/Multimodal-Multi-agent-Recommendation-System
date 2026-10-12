from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field


class WebSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    snippet: str
    source: Literal["tavily"] = "tavily"
    published_at: str | None = None


class RestaurantWebEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    location: str | None = None
    cuisine: str | None = None
    price: str | None = None
    rating: float | None = None
    opening_hours: str | None = None
    menu_items: list[str] = Field(default_factory=list)
    description: str | None = None
    source_url: str
    source: Literal["tavily"] = "tavily"
    published_at: str | None = None

    @classmethod
    def from_search_result(cls, result: WebSearchResult) -> "RestaurantWebEvidence":
        return cls(
            description=result.snippet or None,
            source_url=result.url,
            published_at=result.published_at,
        )


class TavilyWebSearchClient:
    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        *,
        post: Callable | None = None,
        timeout: float = 10.0,
    ) -> None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required to construct TavilyWebSearchClient")
        self.api_key = api_key
        self.timeout = timeout
        if post is None:
            import requests

            post = requests.post
        self._post = post

    def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        response = self._post(
            self.endpoint,
            json={"api_key": self.api_key, "query": query, "max_results": max_results},
            timeout=self.timeout,
        )
        payload = response.json() if hasattr(response, "json") else response
        output: list[WebSearchResult] = []
        for item in payload.get("results", []):
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("content") or item.get("snippet") or "").strip()
            if not url and not snippet:
                continue
            output.append(
                WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    published_at=item.get("published_date"),
                )
            )
            if len(output) >= max_results:
                break
        return output
