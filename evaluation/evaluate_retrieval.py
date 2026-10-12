from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Settings
from src.retrieval.fusion import fuse_rank
from src.retrieval.vector_store import VectorStore


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    _validate_k(k)
    if not ranked or not relevant:
        return 0.0
    return sum(item in relevant for item in ranked[:k]) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    _validate_k(k)
    if not relevant:
        return 0.0
    return sum(item in relevant for item in ranked[:k]) / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    _validate_k(k)
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    _validate_k(k)
    return float(any(item in relevant for item in ranked[:k]))


def validate_queries(queries: list[dict]) -> None:
    qids = [item.get("qid") for item in queries]
    if len(qids) != len(set(qids)):
        raise ValueError("duplicate query IDs are not allowed")
    for item in queries:
        if item.get("split") not in {"dev", "heldout"}:
            raise ValueError("query split must be dev or heldout")
        if not isinstance(item.get("relevant_ids"), list):
            raise ValueError("relevant_ids must be a list")


def summarize_rankings(rows: list[tuple[list[str], set[str]]], k: int = 5) -> dict[str, float]:
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "hit_rate": 0.0}
    return {
        "precision": sum(precision_at_k(r, g, k) for r, g in rows) / len(rows),
        "recall": sum(recall_at_k(r, g, k) for r, g in rows) / len(rows),
        "mrr": sum(mrr_at_k(r, g, k) for r, g in rows) / len(rows),
        "hit_rate": sum(hit_rate_at_k(r, g, k) for r, g in rows) / len(rows),
    }


def run_benchmark(store: VectorStore, queries: list[dict], k: int = 5) -> dict:
    validate_queries(queries)
    heldout = [item for item in queries if item["split"] == "heldout"]
    by_method = {"text": [], "visual": [], "fused": []}
    traces = []
    for item in heldout:
        kind = item.get("kind", "restaurants")
        prefs = item.get("preferences", {})
        query = item["query"]
        text = store.search(kind, query, prefs, k=k, modality="text")
        visual = store.search(kind, query, prefs, k=k, modality="image")
        fused = fuse_rank(text, visual)[:k]
        relevant = {str(value) for value in item["relevant_ids"]}
        rankings = {
            "text": [str(r["id"]) for r in text],
            "visual": [str(r["id"]) for r in visual],
            "fused": [str(r["id"]) for r in fused],
        }
        for method, ranking in rankings.items():
            by_method[method].append((ranking, relevant))
        traces.append({"qid": item["qid"], "rankings": rankings, "relevant_ids": sorted(relevant)})
    return {
        "split": "heldout",
        "k": k,
        "query_count": len(heldout),
        "methods": {method: summarize_rankings(rows, k) for method, rows in by_method.items()},
        "traces": traces,
        "note": "Small held-out benchmark for project comparison; not a production-quality claim.",
    }


def main() -> int:
    queries = json.loads((REPO_ROOT / "evaluation/queries.json").read_text(encoding="utf-8"))
    validate_queries(queries)
    settings = Settings(api_key="evaluation-does-not-use-llm")
    try:
        store = VectorStore.open(settings.chroma_path)
    except (FileNotFoundError, ImportError) as exc:
        print(f"Benchmark not run: {exc}")
        print("Build the real local index first; no retrieval metrics are fabricated.")
        return 0
    report = run_benchmark(store, queries, k=5)
    output = REPO_ROOT / "evaluation/results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("method   precision  recall  mrr  hit_rate")
    for method, metrics in report["methods"].items():
        print(
            f"{method:7}  {metrics['precision']:.3f}      {metrics['recall']:.3f}   "
            f"{metrics['mrr']:.3f}  {metrics['hit_rate']:.3f}"
        )
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
