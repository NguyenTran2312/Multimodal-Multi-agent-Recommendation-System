from __future__ import annotations

from collections.abc import Callable
from typing import Any


class TextEmbedder:
    def __init__(self, model_name: str, model_factory: Callable[[str], Any] | None = None):
        self.model_name = model_name
        self._model_factory = model_factory or self._default_factory
        self._model = None

    @staticmethod
    def _default_factory(model_name: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)

    @property
    def model(self):
        if self._model is None:
            self._model = self._model_factory(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True)
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return [list(map(float, vector)) for vector in vectors]
