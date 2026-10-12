from __future__ import annotations

import json
from pathlib import Path

RECIPE_NLG_SOURCE = "mbien/recipe_nlg"
VI_REVIEW_SOURCE = "pqthinh232/vietnamese-restaurant-review-sentiment-dataset"


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_recipenlg_row(row: dict, *, source_id: str) -> dict:
    return {
        "id": f"recipenlg_{source_id}",
        "name": row.get("title") or row.get("name") or "",
        "cuisine": None,
        "servings": None,
        "prep_time": None,
        "cook_time": None,
        "total_time": None,
        "ingredients": _as_list(row.get("ingredients")),
        "directions": _as_list(row.get("directions")),
        "image_description": None,
        "source_dataset": RECIPE_NLG_SOURCE,
        "source_id": str(source_id),
        "source_url": row.get("link") or row.get("url"),
    }


def normalize_vi_review_row(row: dict, *, source_id: str, split: str | None = None) -> dict:
    return {
        "reviewId": f"vi_review_{source_id}",
        "userId": None,
        "itemId": None,
        "title": None,
        "text": row.get("review") or row.get("text") or row.get("content") or "",
        "date": None,
        "rating": None,
        "language": "vi",
        "sentiment": row.get("sentiment") or row.get("label"),
        "images": [],
        "image_captions": [],
        "source_dataset": VI_REVIEW_SOURCE,
        "source_id": str(source_id),
        "split": split,
    }


def load_optional_json(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {target}")
    return payload
