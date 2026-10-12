from __future__ import annotations

from src.tools.places import PlaceResult
from src.tools.web_search import WebSearchResult

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


def place_to_candidate(place: PlaceResult) -> dict:
    return {
        "id": f"google_places:{place.provider_id}",
        "record": {
            "name": place.name,
            "location": place.address,
            "cuisine": None,
            "price": place.price_level,
            "rating": place.rating,
            "review_count": place.review_count,
            "website": place.website,
            "opening_hours": place.opening_hours,
            "categories": place.categories,
            "source": place.source,
            "provider_id": place.provider_id,
        },
        "source": place.source,
        "provider_id": place.provider_id,
        "match_type": "external",
    }


def web_result_to_evidence(result: WebSearchResult) -> dict:
    return {
        "title": result.title,
        "text": result.snippet,
        "source": result.source,
        "source_url": result.url,
        "published_at": result.published_at,
    }


def deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    output: list[dict] = []
    for candidate in candidates:
        record = candidate.get("record", {})
        source = str(candidate.get("source") or record.get("source") or "")
        provider_id = str(candidate.get("provider_id") or record.get("provider_id") or "")
        key = (source, provider_id)
        if source and provider_id and key in seen:
            continue
        if source and provider_id:
            seen.add(key)
        output.append(candidate)
    return output
