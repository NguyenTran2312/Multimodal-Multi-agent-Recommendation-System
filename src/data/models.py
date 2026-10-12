from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FoodDocument(BaseModel):
    id: str
    entity_type: Literal["dish", "restaurant", "review", "recipe"]
    language: Literal["vi", "en"] = "vi"
    name: str | None = None
    name_vi: str | None = None
    name_en: str | None = None
    cuisine: str = "Vietnamese"
    dish_tags: list[str] = Field(default_factory=list)
    category_tags: list[str] = Field(default_factory=list)
    description: str = ""
    location: str | None = None
    price_level: Literal["$", "$$", "$$$", "$$$$"] | None = None
    spicy: bool | None = None
    vegetarian: bool | None = None
    ingredients: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    rating: float | None = None
    review_text: str | None = None
    sentiment: dict[str, int] = Field(default_factory=dict)
    image_path: str | None = None
    image_url: str | None = None
    source_dataset: str
    source_id: str
    split: Literal["train", "validation", "test", "unknown"] = "unknown"
