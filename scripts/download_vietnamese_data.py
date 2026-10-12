from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.vietnamese import (
    FOODLENS_SOURCE,
    STREETFOOD_SOURCE,
    VLSP_SOURCE,
    normalize_foodlens_rows,
    normalize_streetfood_row,
    normalize_vlsp_row,
    write_jsonl,
)

STREETFOOD_ARCHIVE = "Vietnamese_StreetFood_14Class.zip"
STREETFOOD_CLASSES = {
    0: "Bánh cuốn", 1: "Bánh giò", 2: "Bánh mì", 3: "Bún bò Huế",
    4: "Bún chả", 5: "Bún đậu mắm tôm", 6: "Bún thịt nướng", 7: "Cháo lòng",
    8: "Cốm", 9: "Cơm tấm", 10: "Cơm rang", 11: "Phở bò", 12: "Gỏi cuốn", 13: "Xôi",
}


def _dataset_splits(dataset):
    column_names = getattr(dataset, "column_names", None)
    if hasattr(dataset, "items") and (column_names is None or isinstance(column_names, dict)):
        yield from dataset.items()
    else:
        yield "train", dataset


def _normalized_split(value: str) -> str:
    lowered = str(value).lower()
    if lowered in {"dev", "valid", "validation"}:
        return "validation"
    if lowered == "test":
        return "test"
    return "train"


def _extract_streetfood(archive: str | Path, raw_dir: Path) -> list:
    target = raw_dir / "vietnamese_streetfood"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(target)

    records = []
    for split_name in ("train", "valid", "validation", "test"):
        for images_dir in target.rglob(f"{split_name}/images"):
            labels_dir = images_dir.parent / "labels"
            for image_path in sorted(
                p for p in images_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ):
                label_path = labels_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    continue
                first = label_path.read_text(encoding="utf-8").strip().split()
                if not first:
                    continue
                try:
                    class_id = int(float(first[0]))
                except ValueError:
                    continue
                name = STREETFOOD_CLASSES.get(class_id)
                if name is None:
                    continue
                row = {"id": str(image_path.relative_to(target)), "label": class_id, "label_name": name}
                records.append(normalize_streetfood_row(row, split=_normalized_split(split_name), image_path=str(image_path)))
    return records


def _materialize_image(row: dict, image_dir: Path, fallback_id: str) -> dict:
    result = dict(row)
    existing = result.get("image_path")
    if existing and Path(str(existing)).exists():
        return result
    image = result.get("image")
    if isinstance(image, str) and Path(image).exists():
        result["image_path"] = image
        return result
    if hasattr(image, "save"):
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"{fallback_id}.jpg"
        image.convert("RGB").save(path, format="JPEG") if hasattr(image, "convert") else image.save(path)
        result["image_path"] = str(path)
    return result


def download_and_normalize(*, load_dataset_fn=None, hf_hub_download_fn=None, output_dir: str | Path = "data/normalized", raw_dir: str | Path = "data/raw") -> dict[str, Path]:
    if load_dataset_fn is None:
        try:
            from datasets import load_dataset as load_dataset_fn
        except ImportError as exc:
            raise RuntimeError("Install dependencies with `pip install -r requirements.txt` before downloading Vietnamese datasets.") from exc
    if hf_hub_download_fn is None:
        try:
            from huggingface_hub import hf_hub_download as hf_hub_download_fn
        except ImportError as exc:
            raise RuntimeError("Install dependencies with `pip install -r requirements.txt` before downloading Vietnamese datasets.") from exc

    output = Path(output_dir)
    raw = Path(raw_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    dishes = []
    reviews = []

    try:
        archive = hf_hub_download_fn(repo_id=STREETFOOD_SOURCE, filename=STREETFOOD_ARCHIVE, repo_type="dataset")
        dishes.extend(_extract_streetfood(archive, raw))

        foodlens = load_dataset_fn(FOODLENS_SOURCE)
        for split, rows in _dataset_splits(foodlens):
            materialized = []
            for index, row in enumerate(rows):
                row = dict(row)
                source_id = str(row.get("image_id") or row.get("id") or index)
                materialized.append(_materialize_image(row, output / "images" / "foodlens", source_id.replace("/", "_")))
            dishes.extend(normalize_foodlens_rows(materialized, split=_normalized_split(split)))

        vlsp = load_dataset_fn(VLSP_SOURCE)
        counter = 0
        for split, rows in _dataset_splits(vlsp):
            for row in rows:
                row = dict(row)
                source_id = str(row.get("id") or counter)
                row_split = _normalized_split(row.get("type") or split)
                reviews.append(normalize_vlsp_row(row, source_id=source_id, split=row_split))
                counter += 1
    except Exception as exc:
        raise RuntimeError(f"Could not download/normalize Vietnamese Hugging Face datasets: {exc}") from exc

    return {
        "dishes": write_jsonl(output / "dishes.jsonl", dishes),
        "reviews": write_jsonl(output / "reviews_vi.jsonl", reviews),
    }


def main() -> None:
    written = download_and_normalize()
    for kind, path in written.items():
        print(f"Wrote {kind}: {path}")


if __name__ == "__main__":
    main()
