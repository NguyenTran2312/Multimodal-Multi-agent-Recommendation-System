from src.embeddings.image import ImageEmbedder
from src.embeddings.text import TextEmbedder


class FakeTextModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=True):
        self.calls.append((list(texts), normalize_embeddings))
        return [[float(len(text)), 1.0] for text in texts]


class FakeImageBackend:
    def __init__(self):
        self.image_calls = []
        self.text_calls = []

    def encode_images(self, images):
        self.image_calls.append(list(images))
        return [[float(i), 2.0] for i, _ in enumerate(images, start=1)]

    def encode_text(self, text):
        self.text_calls.append(text)
        return [float(len(text)), 3.0]


def test_text_embedder_forwards_and_reuses_lazy_model():
    created = []

    def factory(_name):
        model = FakeTextModel()
        created.append(model)
        return model

    embedder = TextEmbedder("fake", model_factory=factory)
    assert embedder.embed_texts([]) == []
    assert embedder.embed_texts(["hi", "food"]) == [[2.0, 1.0], [4.0, 1.0]]
    assert embedder.embed_texts(["again"]) == [[5.0, 1.0]]
    assert len(created) == 1
    assert created[0].calls == [(["hi", "food"], True), (["again"], True)]


def test_image_embedder_forwards_images_and_text_and_reuses_backend():
    created = []

    def factory(_name):
        backend = FakeImageBackend()
        created.append(backend)
        return backend

    embedder = ImageEmbedder("fake", backend_factory=factory)
    assert embedder.embed_images([]) == []
    assert embedder.embed_images(["img-a", "img-b"]) == [[1.0, 2.0], [2.0, 2.0]]
    assert embedder.embed_text("cozy ramen") == [10.0, 3.0]
    assert embedder.embed_text("sushi") == [5.0, 3.0]
    assert len(created) == 1
    assert created[0].image_calls == [["img-a", "img-b"]]
    assert created[0].text_calls == ["cozy ramen", "sushi"]


def test_clip_backend_truncates_text_and_unwraps_pooler_output():
    from types import SimpleNamespace
    from src.embeddings.image import _CLIPBackend

    class Tensor:
        def norm(self, **kwargs): return self
        def clamp_min(self, value): return self
        def __truediv__(self, other): return self
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return [[1.0, 0.0]]

    calls = {}
    backend = object.__new__(_CLIPBackend)
    backend.processor = lambda **kwargs: calls.update(kwargs) or {"input_ids": [1]}
    backend.model = SimpleNamespace(get_text_features=lambda **kwargs: SimpleNamespace(pooler_output=Tensor()))
    backend._torch = SimpleNamespace(no_grad=lambda: __import__('contextlib').nullcontext())
    assert backend.encode_text("x" * 1000) == [1.0, 0.0]
    assert calls["truncation"] is True and calls["max_length"] == 77 and calls["padding"] is True
