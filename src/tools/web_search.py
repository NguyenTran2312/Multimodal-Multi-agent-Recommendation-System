from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str
    published_at: str | None = None


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
                    source="tavily",
                    published_at=item.get("published_date"),
                )
            )
            if len(output) >= max_results:
                break
        return output
