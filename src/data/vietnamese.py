from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from src.data.models import FoodDocument
from src.data.preprocessing import clean_string

STREETFOOD_SOURCE = "HoagMin/Vietnamese_StreetFood_Dataset"
FOODLENS_SOURCE = "Tamir39/foodlensvn"
VLSP_SOURCE = "phuongnt101/VLSP2018-ABSA-Restaurant"

ALIASES: dict[str, str] = {
    "banh mi": "Bánh mì",
    "bun bo hue": "Bún bò Huế",
    "bun cha": "Bún chả",
    "bun thit nuong": "Bún thịt nướng",
    "com tam": "Cơm tấm",
    "goi cuon": "Gỏi cuốn",
    "pho": "Phở",
    "pho bo": "Phở bò",
    "banh xeo": "Bánh xèo",
    "mi quang": "Mì Quảng",
    "hu tieu": "Hủ tiếu",
}
ENGLISH_ALIASES: dict[str, str] = {
    "Bánh mì": "Vietnamese baguette sandwich",
    "Bún bò Huế": "Hue-style spicy beef noodle soup",
    "Bún chả": "grilled pork with rice vermicelli",
    "Bún thịt nướng": "grilled pork vermicelli",
    "Cơm tấm": "broken rice with grilled pork",
    "Gỏi cuốn": "fresh spring rolls",
    "Phở": "Vietnamese noodle soup",
    "Phở bò": "Vietnamese beef noodle soup",
    "Bánh xèo": "Vietnamese savory pancake",
    "Mì Quảng": "Quang-style turmeric noodles",
    "Hủ tiếu": "Southern Vietnamese noodle soup",
}

def _slug(value: object) -> str:
    text = clean_string(str(value or "")).lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"[\s_]+", "_", text).strip("_") or "unknown"


def _dish_name(value: object) -> str:
    raw = clean_string(str(value or ""))
    canonical = {name.casefold(): name for name in ALIASES.values()}
    if raw.casefold() in canonical:
        return canonical[raw.casefold()]
    key = raw.lower().replace("_", "-").replace("-", " ")
    key = re.sub(r"\s+", " ", key)
    return ALIASES.get(key, raw.replace("_", " ").strip().title())


def normalize_streetfood_row(row: dict, *, split: str, image_path: str | None = None) -> FoodDocument:
    label = row.get("label_name") or row.get("class_name") or row.get("dish") or row.get("label") or "unknown"
    name_vi = _dish_name(label)
    source_id = str(row.get("id") or row.get("image_id") or row.get("filename") or _slug(label))
    return FoodDocument(
        id=f"streetfood_{_slug(source_id)}",
        entity_type="dish",
        name=name_vi,
        name_vi=name_vi,
        name_en=ENGLISH_ALIASES.get(name_vi),
        dish_tags=[name_vi] + ([ENGLISH_ALIASES[name_vi]] if name_vi in ENGLISH_ALIASES else []),
        category_tags=["street food"],
        description=f"{name_vi}, a Vietnamese street-food dish.",
        image_path=image_path,
        source_dataset=STREETFOOD_SOURCE,
        source_id=source_id,
        split=split if split in {"train", "validation", "test"} else "unknown",
    )


def _spicy_signal(question: str, answer: str) -> bool | None:
    q = f"{question}".lower()
    a = f"{answer}".lower()
    if not any(token in q for token in ("cay", "spicy", "ớt")):
        return None
    negative = ("không cay", "không có vị cay", "not spicy", "không,", "không.")
    positive = ("có cay", "rất cay", "vị cay", "spicy", "cay")
    if any(token in a for token in negative):
        return False
    if any(token in a for token in positive):
        return True
    return None


def normalize_foodlens_rows(rows: list[dict], *, split: str) -> list[FoodDocument]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = str(row.get("image_id") or row.get("id") or row.get("image") or f"row-{index}")
        grouped[key].append(row)
    output: list[FoodDocument] = []
    for source_id, group in grouped.items():
        first = group[0]
        name_vi = _dish_name(first.get("dish") or first.get("dish_name") or first.get("label") or "unknown")
        signals = {
            signal
            for row in group
            if (signal := _spicy_signal(str(row.get("question", "")), str(row.get("answer", "")))) is not None
        }
        spicy = next(iter(signals)) if len(signals) == 1 else None
        qa_text = " ".join(
            clean_string(f"{row.get('question', '')} {row.get('answer', '')}") for row in group
        ).strip()
        tags = [name_vi]
        if name_vi in ENGLISH_ALIASES:
            tags.append(ENGLISH_ALIASES[name_vi])
        if spicy is True:
            tags.append("spicy")
        output.append(FoodDocument(
            id=f"foodlens_{_slug(source_id)}",
            entity_type="dish",
            name=name_vi,
            name_vi=name_vi,
            name_en=ENGLISH_ALIASES.get(name_vi),
            dish_tags=tags,
            description=qa_text or f"{name_vi}, a Vietnamese dish.",
            spicy=spicy,
            image_path=str(first.get("image_path") or first.get("image") or "") or None,
            source_dataset=FOODLENS_SOURCE,
            source_id=source_id,
            split=split if split in {"train", "validation", "test"} else "unknown",
        ))
    return output


def normalize_vlsp_row(row: dict, *, source_id: str, split: str) -> FoodDocument:
    review_text = clean_string(str(row.get("Review") or row.get("review") or row.get("text") or row.get("sentence") or row.get("content") or ""))
    sentiment: dict[str, int] = {}
    for key, value in row.items():
        if key in {"Review", "review", "text", "sentence", "content", "type", "dataset"}:
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            sentiment[str(key)] = value
    return FoodDocument(
        id=f"vlsp_{_slug(source_id)}",
        entity_type="review",
        description=review_text,
        review_text=review_text,
        sentiment=sentiment,
        source_dataset=VLSP_SOURCE,
        source_id=str(source_id),
        split=split if split in {"train", "validation", "test"} else "unknown",
    )


def _as_dict(record: FoodDocument | dict) -> dict:
    return record.model_dump() if isinstance(record, FoodDocument) else dict(record)


def build_text_document(record: FoodDocument | dict) -> str:
    data = _as_dict(record)
    parts: list[str] = []
    for key in ("name", "name_vi", "name_en", "cuisine"):
        if data.get(key):
            parts.append(str(data[key]))
    for key in ("dish_tags", "category_tags", "ingredients"):
        value = data.get(key) or []
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
    for key in ("description", "review_text"):
        if data.get(key):
            parts.append(str(data[key]))
    return clean_string(" | ".join(parts))


def build_clip_text(record: FoodDocument | dict) -> str:
    data = _as_dict(record)
    parts = [data.get("name_vi") or data.get("name"), data.get("name_en")]
    tags = data.get("dish_tags") or []
    parts.extend(tags[:5] if isinstance(tags, list) else [])
    return clean_string(", ".join(str(item) for item in parts if item))


def write_jsonl(path: str | Path, records: Iterable[FoodDocument]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
    return target


def load_jsonl(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
