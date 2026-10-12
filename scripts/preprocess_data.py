from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

from src.data.preprocessing import preprocess_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize multimodal recommendation datasets.")
    parser.add_argument("--restaurants", default="data/restaurants.json")
    parser.add_argument("--recipes", default="data/recipes.json")
    parser.add_argument("--reviews", default="data/reviews.json")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    preprocess_datasets(args.restaurants, args.recipes, args.reviews, args.output_dir)


if __name__ == "__main__":
    main()
