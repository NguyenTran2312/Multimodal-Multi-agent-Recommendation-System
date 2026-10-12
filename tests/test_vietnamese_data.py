from src.data.models import FoodDocument
from src.data.vietnamese import (
    build_clip_text, build_text_document, load_jsonl, normalize_foodlens_rows,
    normalize_streetfood_row, normalize_vlsp_row, write_jsonl,
)


def test_food_document_defaults_are_independent_and_unicode_preserved():
    a = FoodDocument(id="1", entity_type="dish", name="Bún bò Huế", source_dataset="x", source_id="1")
    b = FoodDocument(id="2", entity_type="dish", source_dataset="x", source_id="2")
    a.dish_tags.append("cay")
    assert b.dish_tags == []
    assert a.name == "Bún bò Huế"


def test_streetfood_does_not_fabricate_restaurant_metadata():
    doc = normalize_streetfood_row({"id": "42", "dish": "com_tam"}, split="train")
    assert doc.name_vi == "Cơm tấm"
    assert doc.price_level is None and doc.location is None and doc.rating is None


def test_foodlens_spicy_aggregation_is_source_supported():
    docs = normalize_foodlens_rows([
        {"image_id": "pho1", "dish": "pho", "question": "Món này có cay không?", "answer": "Không cay."},
        {"image_id": "bun1", "dish": "bun_bo_hue", "question": "Món này có cay không?", "answer": "Có cay."},
        {"image_id": "mix", "dish": "pho", "question": "Món này có cay không?", "answer": "Có cay."},
        {"image_id": "mix", "dish": "pho", "question": "Món này có cay không?", "answer": "Không cay."},
    ], split="train")
    by_id = {d.source_id: d for d in docs}
    assert by_id["pho1"].spicy is False
    assert by_id["bun1"].spicy is True
    assert by_id["mix"].spicy is None


def test_vlsp_preserves_text_and_integer_aspects():
    doc = normalize_vlsp_row({"Review": "", "text": "Đồ ăn ngon nhưng giá cao", "FOOD#QUALITY": 1, "RESTAURANT#PRICES": 2}, source_id="7", split="test")
    assert doc.review_text == "Đồ ăn ngon nhưng giá cao"
    assert doc.sentiment == {"FOOD#QUALITY": 1, "RESTAURANT#PRICES": 2}
    assert doc.rating is None


def test_representations_and_jsonl_roundtrip(tmp_path):
    doc = FoodDocument(id="d1", entity_type="dish", name="Bún bò Huế", name_vi="Bún bò Huế", name_en="Hue beef noodle soup", dish_tags=["spicy", "noodle", "beef", "lemongrass", "street food", "extra"], description="Món nước cay.", source_dataset="x", source_id="1")
    text = build_text_document(doc)
    clip = build_clip_text(doc)
    assert "Món nước cay" in text and "Hue beef noodle soup" in text
    assert len(clip) < len(text)
    path = write_jsonl(tmp_path / "data.jsonl", [doc])
    loaded = load_jsonl(path)
    assert loaded[0]["name"] == "Bún bò Huế"
    assert loaded[0]["source_dataset"] == "x"



def test_downloader_uses_exact_approved_dataset_names(tmp_path):
    import zipfile
    from scripts.download_vietnamese_data import download_and_normalize
    archive = tmp_path / "street.zip"
    image_dir = tmp_path / "zipdata" / "dataset" / "train" / "images"
    label_dir = tmp_path / "zipdata" / "dataset" / "train" / "labels"
    image_dir.mkdir(parents=True); label_dir.mkdir(parents=True)
    (image_dir / "sample.jpg").write_bytes(b"fake")
    (label_dir / "sample.txt").write_text("9 0.5 0.5 1 1", encoding="utf-8")
    with zipfile.ZipFile(archive, "w") as z:
        z.write(image_dir / "sample.jpg", "dataset/train/images/sample.jpg")
        z.write(label_dir / "sample.txt", "dataset/train/labels/sample.txt")
    calls = []
    def fake_load(name):
        calls.append(name)
        if "foodlensvn" in name:
            return {"train": [{"image_id": "p1", "dish": "pho", "question": "Món này có cay không?", "answer": "Không cay."}]}
        return {"train": [{"Review": "Đồ ăn ngon", "FOOD#QUALITY": 1, "type": "test"}]}
    hub_calls = []
    def fake_hub(**kwargs):
        hub_calls.append(kwargs)
        return str(archive)
    written = download_and_normalize(load_dataset_fn=fake_load, hf_hub_download_fn=fake_hub, output_dir=tmp_path / "out", raw_dir=tmp_path / "raw")
    assert calls == ["Tamir39/foodlensvn", "phuongnt101/VLSP2018-ABSA-Restaurant"]
    assert hub_calls[0]["repo_id"] == "HoagMin/Vietnamese_StreetFood_Dataset"
    dishes = load_jsonl(written["dishes"])
    reviews = load_jsonl(written["reviews"])
    assert any(d["name_vi"] == "Cơm tấm" and d["image_path"] for d in dishes)
    assert reviews[0]["review_text"] == "Đồ ăn ngon" and reviews[0]["split"] == "test"
