from __future__ import annotations

from src.tools.web_search import RestaurantWebEvidence, WebSearchResult

FRESHNESS_TERMS = (
    "current",
    "today",
    "open now",
    "nearby",
    "real restaurant",
    "recent",
    "hiện tại",
    "hôm nay",
    "đang mở",
    "gần đây",
    "quán thật",
    "nhà hàng thật",
)


def should_use_external_search(query: str, local_candidates: list[dict], preferences) -> bool:
    lowered = query.casefold()
    if any(term in lowered for term in FRESHNESS_TERMS):
        return True
    return not any(item.get("match_type") == "exact" for item in local_candidates)


def web_result_to_evidence(result: WebSearchResult) -> RestaurantWebEvidence:
    return RestaurantWebEvidence.from_search_result(result)
