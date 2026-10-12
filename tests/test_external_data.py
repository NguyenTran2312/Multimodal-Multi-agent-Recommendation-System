import json

from scripts.download_external_data import download_external_data
from src.data.external import normalize_recipenlg_row, normalize_vi_review_row


def test_recipenlg_normalizes_without_invented_fields():
    row = {
        "title": "Pho",
        "ingredients": ["noodles", "beef"],
        "directions": ["Boil"],
        "link": "https://r",
    }
    out = normalize_recipenlg_row(row, source_id="42")
    assert out["name"] == "Pho"
    assert out["ingredients"] == ["noodles", "beef"]
    assert out["directions"] == ["Boil"]
    assert out["cuisine"] is None
    assert out["prep_time"] is None
    assert out["source_id"] == "42"


def test_vi_review_keeps_unknown_linkage_null():
    out = normalize_vi_review_row(
        {"review": "Ngon", "sentiment": "positive"},
        source_id="7",
        split="train",
    )
    assert out["text"] == "Ngon"
    assert out["language"] == "vi"
    assert out["itemId"] is None
    assert out["rating"] is None
    assert out["sentiment"] == "positive"


def test_download_external_data_uses_fake_loader_and_limits_rows(tmp_path):
    class DatasetDict(dict):
        @property
        def column_names(self):
            return {key: [] for key in self}

    def loader(name):
        if "recipe" in name.lower():
            return DatasetDict(train=[
                {"title": "A", "ingredients": ["x"], "directions": ["y"]},
                {"title": "B", "ingredients": [], "directions": []},
            ])
        return DatasetDict(train=[
            {"review_id": "r1", "review": "Ngon", "sentiment": "positive"},
            {"review_id": "r2", "review": "Tệ", "sentiment": "negative"},
        ])

    recipe_path, review_path = download_external_data(
        output_dir=tmp_path,
        recipe_limit=1,
        review_limit=1,
        load_dataset_fn=loader,
    )
    recipes = json.loads(recipe_path.read_text(encoding="utf-8"))
    reviews = json.loads(review_path.read_text(encoding="utf-8"))
    assert len(recipes) == 1
    assert len(reviews) == 1
    assert recipes[0]["name"] == "A"
    assert reviews[0]["itemId"] is None
