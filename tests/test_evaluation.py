import pytest

from evaluation.evaluate_retrieval import (
    hit_rate_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
    validate_queries,
)


def test_metrics_on_known_ranking():
    ranked = ["a", "b", "c"]
    relevant = {"b", "d"}
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert mrr_at_k(ranked, relevant, 3) == 0.5
    assert hit_rate_at_k(ranked, relevant, 1) == 0.0
    assert hit_rate_at_k(ranked, relevant, 2) == 1.0


def test_metrics_handle_empty_relevance():
    assert precision_at_k([], set(), 3) == 0.0
    assert recall_at_k(["a"], set(), 3) == 0.0
    assert mrr_at_k(["a"], set(), 3) == 0.0
    assert hit_rate_at_k(["a"], set(), 3) == 0.0


def test_validate_queries_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate query IDs"):
        validate_queries([
            {"qid": "q1", "split": "dev", "query": "a", "relevant_ids": ["1"]},
            {"qid": "q1", "split": "heldout", "query": "b", "relevant_ids": ["2"]},
        ])


def test_vietnamese_heldout_group_covers_required_cases():
    import json
    from pathlib import Path
    queries = json.loads(Path("evaluation/queries.json").read_text(encoding="utf-8"))
    vi = [q for q in queries if q.get("group") == "vietnamese"]
    assert all(q["split"] == "heldout" for q in vi)
    assert {q.get("case") for q in vi} == {
        "exact_dish", "alias", "spicy", "price_fallback", "genuine_no_match", "dish_assisted_restaurant"
    }
    assert all(q["relevant_ids"] == [] for q in vi)
