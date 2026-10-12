from __future__ import annotations

import json
from pathlib import Path


def _load_json_list(path: str | Path) -> list[dict]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Data file not found: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {resolved}: {exc.msg}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {resolved}")
    return payload


def load_restaurants(path: str | Path) -> list[dict]:
    return _load_json_list(path)


def load_recipes(path: str | Path) -> list[dict]:
    return _load_json_list(path)


def load_reviews(path: str | Path) -> list[dict]:
    return _load_json_list(path)
