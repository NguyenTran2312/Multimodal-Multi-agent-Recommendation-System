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
    retrieval_k: int = 3
    candidate_depth: int = 20
    tavily_api_key: str | None = None
    enable_web_search: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured. Copy .env.example to .env "
                "and add your API key."
            )

        def _optional_key(name: str) -> str | None:
            value = os.getenv(name, "").strip()
            return value or None

        def _flag(name: str) -> bool:
            return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            api_key=api_key,
            tavily_api_key=_optional_key("TAVILY_API_KEY"),
            enable_web_search=_flag("ENABLE_WEB_SEARCH"),
        )
