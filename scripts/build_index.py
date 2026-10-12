from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Settings
from src.data.loader import load_recipes, load_restaurants
from src.data.vietnamese import load_jsonl
from src.embeddings.image import ImageEmbedder
from src.embeddings.text import TextEmbedder
from src.retrieval.vector_store import VectorStore, index_food_documents, index_records


def build_index(settings: Settings | None = None) -> VectorStore:
    settings = settings or Settings(api_key="index-build-does-not-use-llm")
    store = VectorStore.create(
        settings.chroma_path,
        text_embedder=TextEmbedder(settings.text_model_name),
        image_embedder=ImageEmbedder(settings.image_model_name),
    )
    index_records(store, "restaurants", load_restaurants(settings.restaurants_path))
    index_records(store, "recipes", load_recipes(settings.recipes_path))
    normalized_dir = Path("data/normalized")
    normalized_records = load_jsonl(normalized_dir / "dishes.jsonl") + load_jsonl(normalized_dir / "reviews_vi.jsonl")
    if normalized_records:
        index_food_documents(store, normalized_records)
    return store


def main() -> None:
    settings = Settings(api_key="index-build-does-not-use-llm")
    build_index(settings)
    print(f"Index written to {settings.chroma_path}")


if __name__ == "__main__":
    main()
