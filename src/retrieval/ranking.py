from __future__ import annotations

from copy import deepcopy


SEMANTIC_WEIGHT = 0.50
DISH_WEIGHT = 0.20
PREFERENCE_WEIGHT = 0.20
REVIEW_WEIGHT = 0.10

_BUDGET_PRICES = {
    "cheap": {"$", "cheap", "budget"},
    "moderate": {"$$", "moderate"},
    "expensive": {"$$$", "$$$$", "expensive", "fine dining"},
}


def _text(record: dict) -> str:
    values: list[str] = []
    for value in record.values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(v) for v in value)
    return " ".join(values).casefold()


def _matches(value, wanted) -> bool:
    if not value or not wanted:
        return False
    values = value if isinstance(value, list) else [value]
    wants = wanted if isinstance(wanted, list) else [wanted]
    return any(str(a).casefold() == str(b).casefold() for a in values for b in wants)


def preference_score(record: dict, preferences: dict, *, dish_terms: list[str] | None = None) -> float:
    components: list[float] = []
    if preferences.get("cuisine"):
        components.append(1.0 if _matches(record.get("cuisine"), preferences["cuisine"]) else 0.0)
    if preferences.get("budget") or preferences.get("price"):
        wanted = preferences.get("price")
        if not wanted:
            wanted = _BUDGET_PRICES.get(str(preferences.get("budget", "")).lower(), {str(preferences.get("budget"))})
        components.append(1.0 if _matches(record.get("price") or record.get("price_level"), list(wanted) if isinstance(wanted, set) else wanted) else 0.0)
    if preferences.get("location"):
        location = str(preferences["location"]).casefold()
        record_location = str(record.get("location") or "").casefold()
        components.append(1.0 if location and record_location and (location in record_location or record_location in location) else 0.0)
    atmosphere = preferences.get("atmosphere") or []
    if atmosphere:
        haystack = _text(record)
        components.append(1.0 if any(str(term).casefold() in haystack for term in atmosphere) else 0.0)
    if not components:
        return 0.0
    return sum(components) / len(components)


def _dish_score(record: dict, dish_terms: list[str] | None) -> float:
    if not dish_terms:
        return 0.0
    haystack = _text(record)
    matches = sum(1 for term in dish_terms if term and str(term).casefold() in haystack)
    return matches / len(dish_terms)


def preference_mismatches(record: dict, preferences: dict) -> list[str]:
    mismatches: list[str] = []
    for key in ("cuisine", "budget", "location", "atmosphere"):
        if preferences.get(key) and preference_score(record, {key: preferences[key]}) < 1.0:
            mismatches.append(key)
    return mismatches


def rerank_candidates(candidates: list[dict], preferences: dict, *, dish_terms: list[str] | None = None, review_scores: dict[str, float] | None = None) -> list[dict]:
    if not candidates:
        return []
    raw_semantic = [max(0.0, float(item.get("fusion_score", 0.0))) for item in candidates]
    high = max(raw_semantic) if raw_semantic else 0.0
    semantic = [score / high if high > 0 else 0.0 for score in raw_semantic]
    reviews = review_scores or {}
    ranked: list[dict] = []
    for item, semantic_score in zip(candidates, semantic):
        out = deepcopy(item)
        record = dict(out.get("record") or {})
        if out.get("document"):
            record["_document"] = out["document"]
        dish_score = _dish_score(record, dish_terms)
        pref_score = preference_score(record, preferences, dish_terms=dish_terms)
        review_score = max(0.0, min(1.0, float(reviews.get(str(out.get("id")), 0.0))))
        score = SEMANTIC_WEIGHT * semantic_score + DISH_WEIGHT * dish_score + PREFERENCE_WEIGHT * pref_score + REVIEW_WEIGHT * review_score
        out["ranking_components"] = {"semantic": semantic_score, "dish": dish_score, "preference": pref_score, "review": review_score}
        out["ranking_score"] = score
        mismatches = preference_mismatches(record, preferences)
        out["match_type"] = "relaxed" if mismatches else "exact"
        out["preference_mismatches"] = mismatches
        ranked.append(out)
    return sorted(ranked, key=lambda item: -item["ranking_score"])
