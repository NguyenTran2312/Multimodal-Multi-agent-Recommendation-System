from __future__ import annotations

from typing import Any


def build_where(preferences: dict[str, Any] | None, *, strict_only: bool = False) -> dict | None:
    preferences = preferences or {}
    source = preferences.get("strict_constraints", {}) if strict_only else preferences
    if strict_only and not isinstance(source, dict):
        source = {}
    clauses: list[dict] = []
    cuisine = source.get("cuisine")
    if cuisine:
        clauses.append({"cuisine": {"$in": cuisine}} if isinstance(cuisine, list) else {"cuisine": {"$eq": cuisine}})
    price = source.get("price")
    if price:
        clauses.append({"price": {"$eq": price}})
    elif source.get("budget"):
        budget = str(source["budget"]).lower()
        mapped = {"cheap": ["$", "cheap", "budget"], "moderate": ["$$", "moderate"], "expensive": ["$$$", "$$$$", "expensive", "fine dining"]}.get(budget, [source["budget"]])
        clauses.append({"price": {"$in": mapped}})
    location = source.get("location")
    if location:
        clauses.append({"location": {"$eq": location}})
    dietary = source.get("dietary_preferences")
    if dietary:
        values = dietary if isinstance(dietary, list) else [dietary]
        clauses.append({"dietary": {"$in": values}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search_restaurants(store, query: str, preferences: dict | None = None, k: int = 5, *, strict_only: bool = False) -> list[dict]:
    return store.search("restaurants", query, preferences or {}, k=k, modality="text", strict_only=strict_only)


def search_recipes(store, query: str, preferences: dict | None = None, k: int = 5, *, strict_only: bool = False) -> list[dict]:
    return store.search("recipes", query, preferences or {}, k=k, modality="text", strict_only=strict_only)


def search_dishes(store, query: str, k: int = 5) -> list[dict]:
    return store.search("dish", query, {}, k=k, modality="text")


def search_reviews(store, query: str, k: int = 5) -> list[dict]:
    return store.search("review", query, {}, k=k, modality="text")
