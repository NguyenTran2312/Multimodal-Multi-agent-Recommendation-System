from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from src.data.loader import load_recipes, load_restaurants, load_reviews


def clean_string(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clean_string(value: str) -> str:
    return clean_string(value)


def _clean_value(value):
    if isinstance(value, str):
        return _clean_string(value)
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_value(item) for key, item in value.items()}
    return value


def _valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def normalize_restaurant(record: dict) -> dict:
    return _clean_value(dict(record))


def normalize_recipe(record: dict) -> dict:
    return _clean_value(dict(record))


def normalize_review(record: dict) -> dict:
    normalized = _clean_value(dict(record))
    images = normalized.get("images", [])
    if isinstance(images, str):
        try:
            parsed = ast.literal_eval(images)
        except (ValueError, SyntaxError):
            parsed = []
        images = parsed if isinstance(parsed, list) else []
    if isinstance(images, list):
        normalized["images"] = [
            image for image in images if isinstance(image, str) and _valid_http_url(image)
        ]
    return normalized


def preprocess_datasets(
    restaurants_path: str | Path,
    recipes_path: str | Path,
    reviews_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {
        "restaurants": [normalize_restaurant(r) for r in load_restaurants(restaurants_path)],
        "recipes": [normalize_recipe(r) for r in load_recipes(recipes_path)],
        "reviews": [normalize_review(r) for r in load_reviews(reviews_path)],
    }
    written: dict[str, Path] = {}
    for name, records in datasets.items():
        path = output / f"{name}.json"
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        written[name] = path
    return written

CUISINE_KEYWORDS = (
    "Italian", "Indian", "Chinese", "Japanese", "Korean", "Vietnamese",
    "Lebanese", "Mediterranean", "French", "Oaxacan", "Mexican", "Tex-Mex",
    "Persian", "Spanish", "Szechuan", "Seafood", "Soul Food", "Sushi",
    "Vegan", "Dim Sum", "Californian", "American", "Thai", "Baja",
)


def parse_restaurant_map(text: str) -> list[dict]:
    records: list[dict] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    for paragraph in paragraphs:
        bold = re.findall(r"\*\*(.+?)\*\*", paragraph)
        if not bold or paragraph.startswith("###"):
            continue
        name = bold[0].strip()
        location_match = re.search(r"(?:\bin\b|\bof\b)\s+\*\*([^*]+)\*\*", paragraph)
        location = location_match.group(1).strip() if location_match else ""
        cuisine = ""
        for token in bold[1:]:
            for keyword in CUISINE_KEYWORDS:
                if keyword.lower() in token.lower():
                    cuisine = keyword
                    break
            if cuisine:
                break
        price_match = re.search(r"Price range:\s*(\${1,4})", paragraph)
        price = price_match.group(1) if price_match else ""
        description = re.sub(r"\*\*([^*]+)\*\*", r"\1", paragraph)
        description = _clean_string(description)
        records.append(
            {
                "id": 1000001 + len(records),
                "name": name,
                "location": location,
                "cuisine": cuisine,
                "price": price,
                "description": description,
            }
        )
    return records
