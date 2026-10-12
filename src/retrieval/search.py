from __future__ import annotations

from typing import Any


def build_where(preferences: dict[str, Any] | None) -> dict | None:
    preferences = preferences or {}
    clauses: list[dict] = []
    cuisine = preferences.get("cuisine")
    if cuisine:
        if isinstance(cuisine, list):
            clauses.append({"cuisine": {"$in": cuisine}})
        else:
            clauses.append({"cuisine": {"$eq": cuisine}})
    price = preferences.get("price")
    if price:
        clauses.append({"price": {"$eq": price}})
    elif preferences.get("budget"):
        budget = str(preferences["budget"]).lower()
        mapped = {
            "cheap": ["$", "cheap", "budget"],
            "moderate": ["$$", "moderate"],
            "expensive": ["$$$", "$$$$", "expensive", "fine dining"],
        }.get(budget, [preferences["budget"]])
        clauses.append({"price": {"$in": mapped}})
    location = preferences.get("location")
    if location:
        clauses.append({"location": {"$eq": location}})
    dietary = preferences.get("dietary_preferences")
    if dietary:
        values = dietary if isinstance(dietary, list) else [dietary]
        clauses.append({"dietary": {"$in": values}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search_restaurants(store, query: str, preferences: dict | None = None, k: int = 5) -> list[dict]:
    return store.search("restaurants", query, preferences or {}, k=k, modality="text")


def search_recipes(store, query: str, preferences: dict | None = None, k: int = 5) -> list[dict]:
    return store.search("recipes", query, preferences or {}, k=k, modality="text")
