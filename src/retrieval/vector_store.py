from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.embeddings.image import ImageEmbedder
from src.embeddings.text import TextEmbedder
from src.retrieval.search import build_where


COLLECTIONS = ("restaurants_text", "restaurants_image", "recipes_text", "recipes_image")


class VectorStore:
    def __init__(self, path: Path, client, text_embedder, image_embedder):
        self.path = path
        self.client = client
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.collections = {name: client.get_collection(name) for name in COLLECTIONS}

    @staticmethod
    def _default_client(path: Path):
        import chromadb

        return chromadb.PersistentClient(path=str(path))

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        client_factory: Callable[[Path], Any] | None = None,
        text_embedder=None,
        image_embedder=None,
    ) -> "VectorStore":
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Vector index not found at {resolved}. Run `python scripts/build_index.py` first."
            )
        factory = client_factory or cls._default_client
        client = factory(resolved)
        return cls(
            resolved,
            client,
            text_embedder or TextEmbedder("sentence-transformers/all-MiniLM-L6-v2"),
            image_embedder or ImageEmbedder("openai/clip-vit-base-patch32"),
        )

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        client_factory: Callable[[Path], Any] | None = None,
        text_embedder=None,
        image_embedder=None,
    ) -> "VectorStore":
        resolved = Path(path)
        resolved.mkdir(parents=True, exist_ok=True)
        factory = client_factory or cls._default_client
        client = factory(resolved)
        for name in COLLECTIONS:
            client.get_or_create_collection(name)
        return cls(
            resolved,
            client,
            text_embedder or TextEmbedder("sentence-transformers/all-MiniLM-L6-v2"),
            image_embedder or ImageEmbedder("openai/clip-vit-base-patch32"),
        )

    def search(
        self,
        kind: str,
        query: str,
        preferences: dict | None,
        *,
        k: int = 5,
        modality: str = "text",
    ) -> list[dict]:
        if kind not in {"restaurants", "recipes"}:
            raise ValueError("kind must be 'restaurants' or 'recipes'")
        if modality not in {"text", "image"}:
            raise ValueError("modality must be 'text' or 'image'")
        if k <= 0:
            raise ValueError("k must be positive")
        collection = self.collections[f"{kind}_{modality}"]
        if modality == "text":
            embedding = self.text_embedder.embed_texts([query])[0]
        else:
            embedding = self.image_embedder.embed_text(query)
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": k,
            "include": ["metadatas", "documents", "distances"],
        }
        where = build_where(preferences)
        if where is not None:
            kwargs["where"] = where
        response = collection.query(**kwargs)
        ids = (response.get("ids") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        return [
            {
                "id": item_id,
                "record": metadata or {},
                "document": document,
                "distance": float(distance),
                "modality": modality,
            }
            for item_id, metadata, document, distance in zip(ids, metadatas, documents, distances)
        ]


def _metadata(record: dict) -> dict:
    allowed = (str, int, float, bool)
    return {key: value for key, value in record.items() if isinstance(value, allowed)}


def _record_text(record: dict) -> str:
    parts = []
    for value in record.values():
        if isinstance(value, (str, int, float)):
            parts.append(str(value))
        elif isinstance(value, list):
            parts.extend(str(item) for item in value if isinstance(item, (str, int, float)))
    return " | ".join(parts)


def index_records(store: VectorStore, kind: str, records: list[dict]) -> None:
    if not records:
        return
    ids = [str(record.get("itemId") or record.get("id") or index) for index, record in enumerate(records)]
    documents = [_record_text(record) for record in records]
    metadata = [_metadata(record) for record in records]
    text_vectors = store.text_embedder.embed_texts(documents)
    store.collections[f"{kind}_text"].upsert(
        ids=ids, embeddings=text_vectors, documents=documents, metadatas=metadata
    )
    visual_texts = [
        str(record.get("image_description") or record.get("image_caption") or document)
        for record, document in zip(records, documents)
    ]
    image_vectors = [store.image_embedder.embed_text(text) for text in visual_texts]
    store.collections[f"{kind}_image"].upsert(
        ids=ids, embeddings=image_vectors, documents=visual_texts, metadatas=metadata
    )
