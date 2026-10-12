from __future__ import annotations

from collections.abc import Callable
from typing import Any


class _CLIPBackend:
    def __init__(self, model_name: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()

    @staticmethod
    def _rows(tensor) -> list[list[float]]:
        normalized = tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return normalized.detach().cpu().tolist()

    def encode_images(self, images: list) -> list[list[float]]:
        inputs = self.processor(images=images, return_tensors="pt")
        with self._torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return self._rows(features)

    def encode_text(self, text: str) -> list[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with self._torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return self._rows(features)[0]


class ImageEmbedder:
    def __init__(self, model_name: str, backend_factory: Callable[[str], Any] | None = None):
        self.model_name = model_name
        self._backend_factory = backend_factory or _CLIPBackend
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            self._backend = self._backend_factory(self.model_name)
        return self._backend

    def embed_images(self, images: list) -> list[list[float]]:
        if not images:
            return []
        vectors = self.backend.encode_images(images)
        return [list(map(float, vector)) for vector in vectors]

    def embed_text(self, text: str) -> list[float]:
        return list(map(float, self.backend.encode_text(text)))
