from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.data.models import FoodDocument
from src.data.vietnamese import build_clip_text, build_text_document
from src.embeddings.image import ImageEmbedder
from src.embeddings.text import TextEmbedder
from src.retrieval.search import build_where

REQUIRED_COLLECTIONS = ("restaurants_text", "restaurants_image", "recipes_text", "recipes_image")
OPTIONAL_COLLECTIONS = ("dish_text", "dish_image", "review_text")
COLLECTIONS = REQUIRED_COLLECTIONS + OPTIONAL_COLLECTIONS
INDEX_BATCH_SIZE = 1000


class VectorStore:
    def __init__(self, path: Path, client, text_embedder, image_embedder, collections: dict[str, Any]):
        self.path = path
        self.client = client
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.collections = collections

    @staticmethod
    def _default_client(path: Path):
        import chromadb
        return chromadb.PersistentClient(path=str(path))

    @classmethod
    def open(cls, path: str | Path, *, client_factory: Callable[[Path], Any] | None = None, text_embedder=None, image_embedder=None) -> "VectorStore":
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Vector index not found at {resolved}. Run `python scripts/build_index.py` first.")
        factory = client_factory or cls._default_client
        client = factory(resolved)
        collections: dict[str, Any] = {}
        for name in REQUIRED_COLLECTIONS:
            collections[name] = client.get_collection(name)
        for name in OPTIONAL_COLLECTIONS:
            try:
                collections[name] = client.get_collection(name)
            except Exception:
                pass
        return cls(resolved, client, text_embedder or TextEmbedder("sentence-transformers/all-MiniLM-L6-v2"), image_embedder or ImageEmbedder("openai/clip-vit-base-patch32"), collections)

    @classmethod
    def create(cls, path: str | Path, *, client_factory: Callable[[Path], Any] | None = None, text_embedder=None, image_embedder=None) -> "VectorStore":
        resolved = Path(path)
        resolved.mkdir(parents=True, exist_ok=True)
        factory = client_factory or cls._default_client
        client = factory(resolved)
        collections = {name: client.get_or_create_collection(name) for name in COLLECTIONS}
        return cls(resolved, client, text_embedder or TextEmbedder("sentence-transformers/all-MiniLM-L6-v2"), image_embedder or ImageEmbedder("openai/clip-vit-base-patch32"), collections)

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def search(self, kind: str, query: str, preferences: dict | None, *, k: int = 5, modality: str = "text", strict_only: bool = False) -> list[dict]:
        aliases = {"dishes": "dish", "reviews": "review", "restaurants": "restaurants", "recipes": "recipes"}
        prefix = aliases.get(kind, kind)
        if prefix == "review" and modality != "text":
            return []
        collection_name = f"{prefix}_{modality}"
        if collection_name not in self.collections:
            return []
        if modality not in {"text", "image"}:
            raise ValueError("modality must be 'text' or 'image'")
        if k <= 0:
            raise ValueError("k must be positive")
        collection = self.collections[collection_name]
        embedding = self.text_embedder.embed_texts([query])[0] if modality == "text" else self.image_embedder.embed_text(query)
        kwargs = {"query_embeddings": [embedding], "n_results": k, "include": ["metadatas", "documents", "distances"]}
        where = build_where(preferences, strict_only=strict_only)
        if where is not None:
            kwargs["where"] = where
        response = collection.query(**kwargs)
        ids = (response.get("ids") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        return [{"id": item_id, "record": metadata or {}, "document": document, "distance": float(distance), "modality": modality} for item_id, metadata, document, distance in zip(ids, metadatas, documents, distances)]


def _metadata(record: dict) -> dict:
    allowed = (str, int, float, bool)
    metadata: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, allowed) and value != "":
            mapped = "price" if key == "price_level" else key
            metadata[mapped] = value
    return metadata


def _record_text(record: dict) -> str:
    parts = []
    for value in record.values():
        if isinstance(value, (str, int, float)):
            parts.append(str(value))
        elif isinstance(value, list):
            parts.extend(str(item) for item in value if isinstance(item, (str, int, float)))
    return " | ".join(parts)


def _clip_record_text(record: dict) -> str:
    preferred = [record.get("name"), record.get("title"), record.get("cuisine"), record.get("description"), record.get("image_description"), record.get("image_caption")]
    return " | ".join(str(v) for v in preferred if v)[:1200]


def _batches(items: list, size: int = INDEX_BATCH_SIZE):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def index_records(store: VectorStore, kind: str, records: list[dict]) -> None:
    if not records:
        return
    for offset in range(0, len(records), INDEX_BATCH_SIZE):
        batch = records[offset : offset + INDEX_BATCH_SIZE]
        ids = [str(record.get("itemId") or record.get("id") or offset + index) for index, record in enumerate(batch)]
        documents = [_record_text(record) for record in batch]
        metadata = [_metadata(record) for record in batch]
        text_vectors = store.text_embedder.embed_texts(documents)
        store.collections[f"{kind}_text"].upsert(ids=ids, embeddings=text_vectors, documents=documents, metadatas=metadata)

        visual_texts = [str(record.get("image_description") or record.get("image_caption") or _clip_record_text(record)) for record in batch]
        image_vectors = [store.image_embedder.embed_text(text) for text in visual_texts]
        store.collections[f"{kind}_image"].upsert(ids=ids, embeddings=image_vectors, documents=visual_texts, metadatas=metadata)


def index_food_documents(store: VectorStore, records: list[dict | FoodDocument]) -> None:
    dishes: list[tuple[str, dict, str, str]] = []
    reviews: list[tuple[str, dict, str]] = []
    data_by_id: dict[str, dict] = {}
    for item in records:
        data = item.model_dump() if isinstance(item, FoodDocument) else dict(item)
        entity_type = data.get("entity_type")
        item_id = str(data.get("id") or data.get("source_id"))
        data_by_id[item_id] = data
        metadata = _metadata(data)
        if entity_type == "dish":
            dishes.append((item_id, metadata, build_text_document(data), build_clip_text(data)))
        elif entity_type == "review":
            reviews.append((item_id, metadata, build_text_document(data)))

    if dishes and store.has_collection("dish_text"):
        for batch in _batches(dishes):
            ids = [x[0] for x in batch]
            docs = [x[2] for x in batch]
            store.collections["dish_text"].upsert(
                ids=ids,
                embeddings=store.text_embedder.embed_texts(docs),
                documents=docs,
                metadatas=[x[1] for x in batch],
            )

    if dishes and store.has_collection("dish_image"):
        from PIL import Image

        for batch in _batches(dishes):
            image_ids, images, image_docs, metas = [], [], [], []
            for item_id, metadata, _, clip_text in batch:
                data = data_by_id.get(item_id, {})
                image_path = data.get("image_path")
                if image_path and Path(image_path).exists():
                    with Image.open(image_path) as image:
                        images.append(image.convert("RGB").copy())
                    image_ids.append(item_id)
                    image_docs.append(clip_text)
                    metas.append(metadata)
            if images:
                vectors = store.image_embedder.embed_images(images)
                store.collections["dish_image"].upsert(
                    ids=image_ids,
                    embeddings=vectors,
                    documents=image_docs,
                    metadatas=metas,
                )

    if reviews and store.has_collection("review_text"):
        for batch in _batches(reviews):
            ids = [x[0] for x in batch]
            docs = [x[2] for x in batch]
            store.collections["review_text"].upsert(
                ids=ids,
                embeddings=store.text_embedder.embed_texts(docs),
                documents=docs,
                metadatas=[x[1] for x in batch],
            )
