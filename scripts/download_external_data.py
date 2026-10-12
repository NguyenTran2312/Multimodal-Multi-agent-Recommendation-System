from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.external import (
    RECIPE_NLG_SOURCE,
    VI_REVIEW_SOURCE,
    normalize_recipenlg_row,
    normalize_vi_review_row,
)


def _take(rows, limit: int):
    if not limit or limit < 1:
        yield from rows
        return
    for index, row in enumerate(rows):
        if index >= limit:
            break
        yield row


def _load_split(load_dataset_fn, name: str, split: str = "train"):
    """Load one split lazily so row limits do not require downloading the full dataset."""
    return load_dataset_fn(name, split=split, streaming=True)


def download_external_data(
    *,
    output_dir: str | Path = "data/external",
    recipe_limit: int = 0,
    review_limit: int = 0,
    load_dataset_fn=None,
) -> tuple[Path, Path]:
    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    recipe_split = "train"
    recipe_rows = _load_split(load_dataset_fn, RECIPE_NLG_SOURCE, recipe_split)
    recipes = [
        normalize_recipenlg_row(dict(row), source_id=f"{recipe_split}-{index}")
        for index, row in enumerate(_take(recipe_rows, recipe_limit))
    ]

    review_split = "train"
    review_rows = _load_split(load_dataset_fn, VI_REVIEW_SOURCE, review_split)
    reviews = [
        normalize_vi_review_row(
            dict(row),
            source_id=str(row.get("review_id") or row.get("id") or f"{review_split}-{index}"),
            split=review_split,
        )
        for index, row in enumerate(_take(review_rows, review_limit))
    ]

    recipe_path = target / "recipes_recipenlg.json"
    review_path = target / "reviews_vi_sentiment.json"
    recipe_path.write_text(json.dumps(recipes, ensure_ascii=False, indent=2), encoding="utf-8")
    review_path.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    return recipe_path, review_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize external culinary datasets")
    parser.add_argument("--output-dir", default="data/external")
    parser.add_argument("--recipe-limit", type=int, default=0)
    parser.add_argument("--review-limit", type=int, default=0)
    args = parser.parse_args()
    recipe_path, review_path = download_external_data(
        output_dir=args.output_dir,
        recipe_limit=args.recipe_limit,
        review_limit=args.review_limit,
    )
    print(f"Wrote {recipe_path}")
    print(f"Wrote {review_path}")


if __name__ == "__main__":
    main()
