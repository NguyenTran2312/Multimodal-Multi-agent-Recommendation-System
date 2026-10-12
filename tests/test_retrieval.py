from pathlib import Path

import pytest

from src.retrieval.search import search_restaurants, search_recipes
from src.retrieval.vector_store import VectorStore


class FakeCollection:
    def __init__(self, response=None):
        self.response = response or {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}
        self.queries = []

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, collections):
        self.collections = collections

    def get_collection(self, name):
        return self.collections[name]


class FakeTextEmbedder:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeImageEmbedder:
    def embed_text(self, text):
        return [0.0, 1.0]


def make_store(tmp_path, restaurant_response=None, recipe_response=None):
    path = tmp_path / "chroma"
    path.mkdir()
    collections = {
        "restaurants_text": FakeCollection(restaurant_response),
        "restaurants_image": FakeCollection(),
        "recipes_text": FakeCollection(recipe_response),
        "recipes_image": FakeCollection(),
    }
    client = FakeClient(collections)
    store = VectorStore.open(
        path,
        client_factory=lambda _: client,
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
    )
    return store, collections


def test_open_missing_index_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"python scripts/build_index.py"):
        VectorStore.open(tmp_path / "missing", client_factory=lambda _: None)


def test_open_existing_index_does_not_build(tmp_path):
    store, _ = make_store(tmp_path)
    assert store.path == tmp_path / "chroma"


def test_restaurant_text_search_returns_records_scores_and_filter(tmp_path):
    response = {
        "ids": [["r1"]],
        "metadatas": [[{"itemId": "r1", "name": "Sakura", "cuisine": "Japanese"}]],
        "documents": [["Cozy Japanese restaurant"]],
        "distances": [[0.2]],
    }
    store, collections = make_store(tmp_path, restaurant_response=response)
    results = search_restaurants(store, "cozy sushi", {"cuisine": "Japanese"}, k=3)
    assert results == [{
        "id": "r1",
        "record": {"itemId": "r1", "name": "Sakura", "cuisine": "Japanese"},
        "document": "Cozy Japanese restaurant",
        "distance": 0.2,
        "modality": "text",
    }]
    query = collections["restaurants_text"].queries[0]
    assert query["query_embeddings"] == [[1.0, 0.0]]
    assert query["n_results"] == 3
    assert query["where"] == {"cuisine": {"$eq": "Japanese"}}


def test_recipe_search_returns_empty_when_filter_removes_everything(tmp_path):
    store, _ = make_store(tmp_path)
    assert search_recipes(store, "vegan soup", {"dietary_preferences": ["vegan"]}, k=5) == []


def test_clip_text_to_image_search_uses_image_collection(tmp_path):
    store, collections = make_store(tmp_path)
    store.search("restaurants", "warm colorful plating", {}, k=2, modality="image")
    query = collections["restaurants_image"].queries[0]
    assert query["query_embeddings"] == [[0.0, 1.0]]
    assert query["n_results"] == 2


def test_old_index_can_open_without_optional_vietnamese_collections(tmp_path):
    store, _ = make_store(tmp_path)
    assert store.has_collection("dish_text") is False


def test_search_missing_optional_collection_returns_empty(tmp_path):
    store, _ = make_store(tmp_path)
    assert store.search("dish", "cơm tấm", {}, k=3) == []


def test_create_builds_optional_collections_and_indexes_normalized_docs(tmp_path):
    from src.data.models import FoodDocument
    from src.retrieval.vector_store import COLLECTIONS, index_food_documents

    class UpsertCollection(FakeCollection):
        def __init__(self):
            super().__init__()
            self.upserts = []
        def upsert(self, **kwargs):
            self.upserts.append(kwargs)

    class CreateClient:
        def __init__(self): self.collections = {}
        def get_or_create_collection(self, name):
            return self.collections.setdefault(name, UpsertCollection())

    class ImageEmbedder(FakeImageEmbedder):
        def embed_images(self, images): return [[0.0, 1.0] for _ in images]

    client = CreateClient()
    store = VectorStore.create(tmp_path / "new-index", client_factory=lambda _: client, text_embedder=FakeTextEmbedder(), image_embedder=ImageEmbedder())
    assert set(store.collections) == set(COLLECTIONS)
    docs = [
        FoodDocument(id="d1", entity_type="dish", name="Cơm tấm", name_vi="Cơm tấm", dish_tags=["broken rice"], sentiment={"x": 1}, source_dataset="x", source_id="d1"),
        FoodDocument(id="v1", entity_type="review", review_text="Đồ ăn ngon", description="Đồ ăn ngon", sentiment={"FOOD#QUALITY": 1}, source_dataset="x", source_id="v1"),
    ]
    index_food_documents(store, docs)
    assert client.collections["dish_text"].upserts[0]["ids"] == ["d1"]
    assert client.collections["review_text"].upserts[0]["ids"] == ["v1"]
    dish_meta = client.collections["dish_text"].upserts[0]["metadatas"][0]
    assert "dish_tags" not in dish_meta and "sentiment" not in dish_meta
