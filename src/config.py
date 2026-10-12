from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_model_name: str = "openai/clip-vit-base-patch32"
    restaurants_path: Path = Path("data/restaurants.json")
    recipes_path: Path = Path("data/recipes.json")
    reviews_path: Path = Path("data/reviews.json")
    chroma_path: Path = Path("data/chroma")
    retrieval_k: int = 5
    candidate_depth: int = 20

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured. Copy .env.example to .env "
                "and add your API key."
            )
        return cls(api_key=api_key)
