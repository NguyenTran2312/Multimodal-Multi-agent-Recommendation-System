from __future__ import annotations

from urllib.parse import urlparse

from src.retrieval.ranking import preference_mismatches, preference_score
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


def restaurant_evidence_to_candidate(
    evidence: RestaurantWebEvidence,
    preferences: dict,
) -> dict | None:
    if not evidence.name or not evidence.source_url:
        return None
    host = urlparse(evidence.source_url).netloc or "web"
    candidate_id = f"tavily:{host}:{abs(hash((evidence.name, evidence.source_url)))}"
    record = evidence.model_dump(exclude_none=True)
    record["source"] = "tavily"
    mismatches = preference_mismatches(record, preferences)
    return {
        "id": candidate_id,
        "record": record,
        "document": evidence.description or evidence.name,
        "source": "tavily",
        "source_url": evidence.source_url,
        "fusion_score": 1.0,
        "ranking_score": 0.5 + 0.5 * preference_score(record, preferences),
        "ranking_components": {
            "semantic": 1.0,
            "dish": 0.0,
            "preference": preference_score(record, preferences),
            "review": 0.0,
        },
        "match_type": "relaxed" if mismatches else "exact",
        "preference_mismatches": mismatches,
    }


def deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    output: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        record = candidate.get("record") or {}
        key = (
            str(record.get("name") or "").casefold(),
            str(record.get("source_url") or candidate.get("source_url") or "").casefold(),
        )
        if key != ("", "") and key in seen:
            continue
        if key != ("", ""):
            seen.add(key)
        output.append(candidate)
    return output
