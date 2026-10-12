from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Settings
from src.data.external import load_optional_json
from src.data.loader import load_recipes, load_restaurants
from src.data.vietnamese import load_jsonl
from src.embeddings.image import ImageEmbedder
from src.embeddings.text import TextEmbedder
from src.retrieval.vector_store import VectorStore, index_food_documents, index_records


def _unique_by_id(records: list[dict]) -> list[dict]:
    output: list[dict] = []
    seen: set[str] = set()
    for record in records:
        item_id = str(record.get("id") or record.get("reviewId") or record.get("source_id") or "")
        if item_id and item_id in seen:
            continue
        if item_id:
            seen.add(item_id)
        output.append(record)
    return output


def _review_food_documents(records: list[dict]) -> list[dict]:
    output: list[dict] = []
    for record in records:
        text = str(record.get("text") or "").strip()
        output.append(
            {
                "id": str(record.get("reviewId") or record.get("source_id") or ""),
                "entity_type": "review",
                "description": text,
                "review_text": text,
                "sentiment": record.get("sentiment"),
                "source_dataset": record.get("source_dataset"),
                "source_id": record.get("source_id"),
                "split": record.get("split") or "unknown",
            }
        )
    return output


def build_index(settings: Settings | None = None) -> VectorStore:
    settings = settings or Settings(api_key="index-build-does-not-use-llm")
    store = VectorStore.create(
        settings.chroma_path,
        text_embedder=TextEmbedder(settings.text_model_name),
        image_embedder=ImageEmbedder(settings.image_model_name),
    )
    index_records(store, "restaurants", load_restaurants(settings.restaurants_path))

    recipes = load_recipes(settings.recipes_path)
    external_dir = Path("data/external")
    recipes += load_optional_json(external_dir / "recipes_recipenlg.json")
    index_records(store, "recipes", _unique_by_id(recipes))

    normalized_dir = Path("data/normalized")
    normalized_records = load_jsonl(normalized_dir / "dishes.jsonl") + load_jsonl(
        normalized_dir / "reviews_vi.jsonl"
    )
    external_reviews = load_optional_json(external_dir / "reviews_vi_sentiment.json")
    normalized_records += _review_food_documents(_unique_by_id(external_reviews))
    if normalized_records:
        index_food_documents(store, normalized_records)
    return store


def main() -> None:
    settings = Settings(api_key="index-build-does-not-use-llm")
    build_index(settings)
    print(f"Index written to {settings.chroma_path}")


if __name__ == "__main__":
    main()
