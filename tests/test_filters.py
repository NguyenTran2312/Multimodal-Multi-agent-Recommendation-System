from src.retrieval.search import build_where


def test_build_where_maps_common_preferences():
    where = build_where({
        "cuisine": "Japanese",
        "budget": "cheap",
        "location": "San Francisco",
        "dietary_preferences": ["vegetarian"],
    })
    assert where == {
        "$and": [
            {"cuisine": {"$eq": "Japanese"}},
            {"price": {"$in": ["$", "cheap", "budget"]}},
            {"location": {"$eq": "San Francisco"}},
            {"dietary": {"$in": ["vegetarian"]}},
        ]
    }


def test_build_where_uses_explicit_price_and_omits_empty_values():
    assert build_where({"price": "$$", "cuisine": ""}) == {"price": {"$eq": "$$"}}


def test_build_where_empty_preferences_returns_none():
    assert build_where({}) is None


def test_build_where_supports_multi_cuisine_preferences():
    assert build_where({"cuisine": ["Japanese", "Korean"]}) == {
        "cuisine": {"$in": ["Japanese", "Korean"]}
    }


def test_strict_only_uses_explicit_strict_constraints_only():
    assert build_where({"budget": "cheap", "strict_constraints": {"location": "District 1"}}, strict_only=True) == {"location": {"$eq": "District 1"}}
    assert build_where({"budget": "cheap"}, strict_only=True) is None
