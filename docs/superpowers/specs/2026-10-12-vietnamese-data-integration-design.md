# Vietnamese Data Integration Design

## Goal

Extend the existing multimodal food recommender with Vietnamese-focused dish and restaurant-review knowledge so Vietnamese queries such as “cheap cơm tấm”, “bún bò Huế”, “bánh mì and Vietnamese coffee”, and “spicy Vietnamese food” produce useful grounded candidates instead of frequent no-match responses.

The change must preserve the current portfolio architecture: local embeddings and Chroma retrieval provide the candidate pool, while LLM agents reason only over retrieved candidates.

## Scope

Use only Vietnamese-focused external datasets identified during research:

- `HoagMin/Vietnamese_StreetFood_Dataset` for Vietnamese dish images and dish labels.
- `Tamir39/foodlensvn` for Vietnamese dish images plus Vietnamese QA-derived attributes such as spiciness.
- `phuongnt101/VLSP2018-ABSA-Restaurant` for Vietnamese restaurant-review text and aspect sentiment.

These datasets enrich dish and review knowledge. They do not provide a trustworthy restaurant directory with consistent identity, address, menu, price, and image information, so actual restaurant recommendation targets remain the curated restaurant catalogue already used by the application.

## Non-goals

- Do not infer or fabricate restaurant identities from review-only datasets.
- Do not run Hugging Face downloads at application startup.
- Do not require Vietnamese datasets for the existing sample application to start.
- Do not add cloud deployment, authentication, or a production database.
- Do not claim retrieval-quality improvements until evaluation is run on held-out queries.

## Normalized data contract

Add `src/data/models.py` with a Pydantic model used before indexing:

```python
class FoodDocument(BaseModel):
    id: str
    entity_type: Literal["dish", "restaurant", "review", "recipe"]
    language: Literal["vi", "en"] = "vi"

    name: str | None = None
    name_vi: str | None = None
    name_en: str | None = None
    cuisine: str = "Vietnamese"

    dish_tags: list[str] = []
    category_tags: list[str] = []
    description: str = ""

    location: str | None = None
    price_level: Literal["$", "$$", "$$$", "$$$$"] | None = None
    spicy: bool | None = None
    vegetarian: bool | None = None

    ingredients: list[str] = []
    instructions: list[str] = []
    rating: float | None = None

    review_text: str | None = None
    sentiment: dict[str, int] = {}

    image_path: str | None = None
    image_url: str | None = None

    source_dataset: str
    source_id: str
    split: Literal["train", "validation", "test", "unknown"] = "unknown"
```

Mutable defaults must use Pydantic-safe default factories in implementation.

## Dataset ingestion

### Street Food dataset

Normalize each image/class row to `entity_type="dish"`.

Populate:

- `name` / `name_vi` from the dish class.
- `name_en` only when a maintained translation map exists.
- `dish_tags` with Vietnamese name plus stable English aliases used for retrieval.
- `category_tags` only from explicit maintained mappings such as `street food`, `noodle dish`, or `rice dish`.
- `image_path` from downloaded local assets.
- `source_dataset`, `source_id`, and `split` from source data.

Do not invent price, restaurant, rating, location, or ingredients.

### FoodLensVN

Aggregate rows by image or dish identifier where multiple QA rows describe the same item.

Populate dish fields plus supported attributes inferred directly from dataset QA, for example:

- `spicy=True/False` only when supported by source answers.
- description fragments based on source answers.

Retain provenance through `source_dataset` and `source_id`.

### VLSP2018-ABSA-Restaurant

Normalize each review to `entity_type="review"`.

Populate:

- `review_text` and `description` with the Vietnamese review.
- aspect sentiment using the original dataset labels.
- `source_dataset`, `source_id`, and `split`.

The sentiment values remain dataset-native values rather than being converted into fabricated star ratings.

## Offline download and normalization

Add `scripts/download_vietnamese_data.py`.

Responsibilities:

1. Use Hugging Face `datasets` to download the three named datasets.
2. Save or materialize raw assets under `data/raw/` where practical.
3. Normalize records using `src/data/vietnamese.py`.
4. Write normalized newline-delimited JSON:

```text
data/normalized/dishes.jsonl
data/normalized/reviews_vi.jsonl
```

Downloaded raw datasets and large image assets should be ignored by Git unless intentionally committing a small sample.

The script must fail with a clear dependency/network error and must never silently replace unavailable data with fabricated examples.

## Embedding representations

### SentenceTransformer text

Create a dedicated text builder instead of indexing every raw field indiscriminately.

For dish records, include:

- Vietnamese and English name when available.
- cuisine.
- dish tags.
- category tags.
- supported description/attributes.

For restaurants, preserve the current descriptive restaurant information and dish/menu tags where available.

For reviews, embed the Vietnamese review text and relevant aspect names.

### CLIP text

Create a separate compact CLIP representation:

```text
<dish Vietnamese name>, <English alias>, <up to five visual/food tags>
```

Do not pass the complete normalized record into CLIP.

CLIP tokenization must always set:

```python
truncation=True
max_length=77
```

The embedding adapter must support Transformers versions where `get_text_features()` / `get_image_features()` may return either a tensor or an output object containing `pooler_output`.

## Chroma layout

Replace the fixed four-collection assumption with these supported collections:

```text
restaurants_text
restaurants_image
recipes_text
recipes_image
dish_text
dish_image
review_text
```

`dish_image` is populated from actual image embeddings when local images are available. When only a textual visual description exists, text-to-CLIP embeddings may be used explicitly and marked through metadata/provenance.

## Chroma metadata

Only scalar filterable values belong in metadata, for example:

```python
{
    "entity_type": "dish",
    "cuisine": "Vietnamese",
    "name": "Bún bò Huế",
    "location": "",
    "price": "",
    "spicy": True,
    "source": "Tamir39/foodlensvn",
}
```

Lists, dictionaries, ingredients, instructions, and full sentiment structures remain in normalized JSON and/or searchable document text rather than being passed directly to Chroma metadata.

## Retrieval architecture

The current strict approach applies cuisine/price/location filters before semantic retrieval. Replace that default behavior with semantic candidate generation followed by preference-aware reranking.

For restaurant intent:

```text
query
  ↓
preference extraction
  ↓
dish semantic retrieval (dish_text + optional dish_image)
  ↓
restaurant semantic retrieval (restaurants_text + restaurants_image)
  ↓
multimodal score fusion
  ↓
preference/dish reranking
  ↓
grounded candidates
```

For recipe intent, preserve the current recipe path while allowing Vietnamese dish signals to help interpret the query in a later extension; V1 does not require dish-to-recipe joins.

## Preference semantics

By default, treat cuisine, budget, atmosphere, and location as soft ranking signals rather than hard filters.

Hard constraints are used only when the user clearly expresses a mandatory constraint, such as:

- “must be under 100,000 VND”
- “only vegetarian”
- “in District 1 only”

V1 may represent explicit strictness with an internal boolean/field derived during preference extraction rather than expanding the public UI.

## Ranking

Add `src/retrieval/ranking.py`.

Use a transparent weighted score with normalized components. Initial provisional weights:

```text
0.50 semantic/multimodal retrieval score
0.20 dish match
0.20 preference match
0.10 review signal
```

Within `preference match`, use approximate weighting:

```text
0.35 cuisine
0.35 dish/menu match
0.20 price
0.10 location
```

These are development defaults only. They must be configurable constants and must not be presented as optimal settings until held-out evaluation supports them.

If review information cannot be connected to a restaurant without fabricated identity, its restaurant-specific contribution is zero. Review knowledge may still support general dish/style understanding, but must not be attributed to a specific restaurant.

## Fallback behavior

If a strict pass returns no candidates:

1. Retry without soft metadata filters.
2. Preserve semantic/dish retrieval.
3. Rerank by closeness to the requested preferences.
4. Return nearby alternatives with an explanation of which preference could not be satisfied.

Example expected behavior:

```text
I couldn't find an exact moderate-price match, but The Neon Noodle Bar is a cheaper spicy option that closely matches the food style you asked for.
```

Do not fabricate a restaurant or dish when no indexed candidate is semantically relevant.

## UI behavior

Keep the Gradio interface thin.

The UI should distinguish:

- exact/strong matches;
- relaxed fallback matches;
- genuine no-match results.

A genuine no-match should mention that no relevant indexed candidate was found, not only that filters failed.

## Backward compatibility

If `data/normalized/dishes.jsonl` and `reviews_vi.jsonl` are absent:

- `scripts/build_index.py` must still build the existing restaurant and recipe collections.
- application startup must still work using those collections.
- Vietnamese-specific collections are optional until the download script has been run.

If the normalized Vietnamese files exist, `build_index.py` indexes them automatically.

## File changes

Add:

```text
src/data/models.py
src/data/vietnamese.py
src/retrieval/ranking.py
scripts/download_vietnamese_data.py
tests/test_vietnamese_data.py
tests/test_ranking.py
```

Modify:

```text
src/data/preprocessing.py
src/embeddings/image.py
src/retrieval/vector_store.py
src/retrieval/search.py
src/retrieval/fusion.py
src/ui/chatbot.py
scripts/build_index.py
tests/test_embeddings.py
tests/test_retrieval.py
requirements.txt
README.md
```

## Testing strategy

Use TDD for all production changes.

Required tests include:

- normalized dish rows preserve provenance and do not invent unsupported fields;
- FoodLens QA aggregation derives supported booleans such as spiciness correctly;
- VLSP aspect labels remain unchanged;
- text and CLIP representation builders produce distinct outputs;
- CLIP long text is truncated to 77 tokens;
- both tensor and `pooler_output` Transformer return forms are supported;
- optional Vietnamese collections do not break an existing index;
- missing normalized Vietnamese files do not break `build_index.py`;
- soft budget mismatch does not eliminate a semantically relevant candidate;
- exact preference matches outrank relaxed alternatives;
- strict constraints remain enforceable;
- fallback results are marked and explained;
- no-result behavior remains grounded and does not fabricate candidates.

All unit tests must run without downloading Hugging Face datasets or calling an LLM/API. Dataset loaders use small fixtures/mocks in tests.

## Evaluation

Extend held-out evaluation with Vietnamese-style queries, separated from any queries used to tune ranking weights.

Include examples covering:

- exact dish match;
- Vietnamese synonym/English alias match;
- spicy preference;
- price mismatch fallback;
- genuine no-match;
- restaurant query where dish semantics improve retrieval.

Report Recall@K, Hit Rate@K, and MRR where qrels exist. Do not report generated recommendation accuracy without labels.

## User workflow

After implementation:

```bash
pip install -r requirements.txt
python scripts/download_vietnamese_data.py
python scripts/build_index.py
python app.py
```

Users who do not want the larger Vietnamese datasets may skip the download command and continue using the bundled sample data.

## Definition of done

The feature is complete when:

1. Existing tests remain green.
2. New Vietnamese ingestion/ranking tests pass without network access.
3. The existing sample app still works without downloaded HF data.
4. Running the download script in a networked environment produces normalized dish/review files from the named Vietnamese datasets.
5. `build_index.py` creates optional dish/review collections when those files are present.
6. Vietnamese food queries can use dish semantics without fabricating restaurant metadata.
7. Soft preference mismatches return ranked alternatives instead of unnecessary empty results.
8. Strict constraints can still produce a genuine no-match.
9. README documents setup, provenance, limitations, and evaluation honestly.
