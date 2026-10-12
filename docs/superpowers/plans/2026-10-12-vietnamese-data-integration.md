# Vietnamese Data Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Vietnamese dish/review knowledge, optional Hugging Face ingestion, richer multimodal indexing, and soft-preference fallback ranking so Vietnamese food queries return grounded alternatives instead of unnecessary no-match results.

**Architecture:** Normalize external Vietnamese sources into a shared `FoodDocument` contract, index compact text/CLIP representations into optional `dish_*` and `review_text` Chroma collections, and keep restaurant/recipe collections backward compatible. Restaurant retrieval remains grounded in the curated restaurant catalogue; Vietnamese dish semantics influence candidate ranking, while budget/cuisine/location are soft signals by default and strict constraints remain enforceable.

**Tech Stack:** Python, Pydantic, Hugging Face `datasets`, SentenceTransformers, Transformers CLIP, ChromaDB, LangChain/OpenAI structured extraction, Gradio, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-vietnamese-data-integration-design.md`

## Global Constraints

- Use only `HoagMin/Vietnamese_StreetFood_Dataset`, `Tamir39/foodlensvn`, and `phuongnt101/VLSP2018-ABSA-Restaurant` as external Vietnamese datasets for this feature.
- Never fabricate restaurant identities, prices, locations, ratings, ingredients, or review-to-restaurant links not supported by source data.
- Hugging Face downloads are offline setup only; application startup must never download datasets.
- Missing Vietnamese normalized files must not break current restaurant/recipe indexing or application startup.
- CLIP text tokenization must always use `truncation=True` and `max_length=77`.
- CLIP feature extraction must accept either raw tensor results or output objects exposing `pooler_output`.
- Lists/dicts stay out of Chroma metadata; only scalar filterable metadata is stored.
- Cuisine, budget, atmosphere, and location are soft ranking signals by default; explicit mandatory constraints remain enforceable.
- Ranking defaults are provisional: semantic/multimodal `0.50`, dish match `0.20`, preference match `0.20`, review signal `0.10`.
- Unit tests must not require network access, dataset downloads, or an LLM/API.
- Do not claim ranking improvements until held-out evaluation supports them.

## Review Focus

- A local Chroma index built before this feature may not contain `dish_text`, `dish_image`, or `review_text`; opening it must still succeed and simply omit Vietnamese-specific retrieval.
- A Vietnamese normalized record can contain Unicode accents, empty optional fields, and source-specific extra data; normalization/indexing must preserve supported values without ASCII loss or fabricated defaults.
- FoodLens rows can disagree or omit spiciness; aggregation must only set `spicy` when source answers support an unambiguous boolean.
- Soft budget/cuisine mismatch must not eliminate a semantically strong restaurant, while a strict constraint must still be able to eliminate it.
- Fallback candidates must be clearly marked and explained, and genuine no-match must not invent a restaurant or dish.

---

## File Structure

**Create**

- `src/data/models.py` — shared normalized `FoodDocument` contract.
- `src/data/vietnamese.py` — dataset-specific normalization, JSONL I/O, embedding-text builders, and provenance mappings.
- `src/retrieval/ranking.py` — transparent soft-preference scoring and fallback reranking.
- `scripts/download_vietnamese_data.py` — explicit Hugging Face download/normalization command.
- `tests/test_vietnamese_data.py` — source normalization and representation tests.
- `tests/test_ranking.py` — ranking, strictness, and fallback tests.

**Modify**

- `src/data/preprocessing.py` — expose/reuse cleaning helpers where needed without changing existing behavior.
- `src/embeddings/image.py` — CLIP truncation and tensor/output compatibility.
- `src/retrieval/vector_store.py` — optional Vietnamese collections, dedicated text/CLIP builders, dish/review indexing, image-file support.
- `src/retrieval/search.py` — distinguish soft vs. strict filters and add dish/review search helpers.
- `src/retrieval/fusion.py` — preserve modality scores in a form consumed by ranking.
- `src/ui/chatbot.py` — Vietnamese dish retrieval, soft reranking, fallback metadata, and user-facing fallback/no-match formatting.
- `scripts/build_index.py` — auto-index normalized Vietnamese files when present.
- `tests/test_embeddings.py` — CLIP compatibility regression tests.
- `tests/test_retrieval.py` — optional collections, soft/strict search, and indexing tests.
- `tests/test_chatbot.py` — fallback and genuine no-match formatting/runtime tests.
- `requirements.txt` — add Hugging Face `datasets`.
- `README.md` — setup, provenance, limitations, and evaluation guidance.

### Task 1: Normalized Vietnamese Data Contract and Source Normalizers

**Files:**
- Create: `src/data/models.py`
- Create: `src/data/vietnamese.py`
- Modify: `src/data/preprocessing.py`
- Test: `tests/test_vietnamese_data.py`

**Interfaces:**
- Produces: `FoodDocument`; `normalize_streetfood_row(row: dict, *, split: str, image_path: str | None = None) -> FoodDocument`; `normalize_foodlens_rows(rows: list[dict], *, split: str) -> list[FoodDocument]`; `normalize_vlsp_row(row: dict, *, source_id: str, split: str) -> FoodDocument`; `build_text_document(record: FoodDocument | dict) -> str`; `build_clip_text(record: FoodDocument | dict) -> str`; `write_jsonl(path: str | Path, records: Iterable[FoodDocument]) -> Path`; `load_jsonl(path: str | Path) -> list[dict]`.
- Consumes: existing whitespace cleaning behavior from `src/data/preprocessing.py`.

- [ ] **Step 1: Write failing model/default/provenance tests**

Add tests proving `FoodDocument` uses independent list/dict defaults, preserves Unicode, and leaves unsupported fields such as `price_level`, `location`, and `rating` as `None` for a street-food row.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `pytest tests/test_vietnamese_data.py -v`

Expected: FAIL because `src.data.models` / `src.data.vietnamese` do not exist.

- [ ] **Step 3: Implement `FoodDocument` with Pydantic-safe default factories**

Use the exact fields and literals from the spec; mutable fields use `Field(default_factory=...)`.

- [ ] **Step 4: Implement Street Food normalization and stable alias/category maps**

Normalize only supported source fields and explicit maintained mappings; do not fill restaurant-specific metadata.

- [ ] **Step 5: Write failing FoodLens aggregation tests**

Cover: repeated QA rows for one image, explicit non-spicy answer -> `False`, explicit spicy answer -> `True`, contradictory/unsupported answers -> `None`, and source provenance retained.

- [ ] **Step 6: Implement `normalize_foodlens_rows(...)`**

Group by source image/dish identifier and derive only unambiguous source-supported booleans.

- [ ] **Step 7: Write failing VLSP preservation tests**

Assert Vietnamese review text is preserved and aspect integer labels are copied unchanged into `sentiment` rather than converted to ratings.

- [ ] **Step 8: Implement `normalize_vlsp_row(...)`**

Recognize aspect-label columns without changing their numeric values.

- [ ] **Step 9: Write failing representation/JSONL tests**

Assert `build_text_document()` contains richer searchable Vietnamese/English aliases and description, while `build_clip_text()` is shorter and capped to name/alias plus at most five tags; JSONL round-trip preserves Unicode/provenance.

- [ ] **Step 10: Implement representation builders and JSONL helpers**

Keep CLIP text compact and deterministic; reuse existing cleaning helpers rather than duplicating whitespace logic.

- [ ] **Step 11: Run Task 1 tests**

Run: `pytest tests/test_vietnamese_data.py tests/test_preprocessing.py -v`

Expected: PASS.

- [ ] **Step 12: Commit**

```bash
git add src/data/models.py src/data/vietnamese.py src/data/preprocessing.py tests/test_vietnamese_data.py
git commit -m "feat: add Vietnamese data normalization"
```

### Task 2: Harden CLIP Embedding Compatibility

**Files:**
- Modify: `src/embeddings/image.py`
- Modify: `tests/test_embeddings.py`

**Interfaces:**
- Produces: `_CLIPBackend.encode_text(text: str) -> list[float]` with 77-token truncation; `_CLIPBackend.encode_images(images: list) -> list[list[float]]` accepting tensor or `.pooler_output` results.
- Consumes: no Task 1 interfaces.

- [ ] **Step 1: Write failing tokenizer regression test**

Use fake processor/model/torch objects and assert text processing is called with `padding=True`, `truncation=True`, `max_length=77`, and `return_tensors="pt"`.

- [ ] **Step 2: Write failing return-shape tests**

Cover both raw tensor-like features and an object with `pooler_output` for text and image feature extraction.

- [ ] **Step 3: Run focused tests and confirm failure**

Run: `pytest tests/test_embeddings.py -v`

Expected: FAIL on missing truncation and output unwrapping behavior.

- [ ] **Step 4: Implement `_unwrap_features(value)` and update CLIP calls**

`_unwrap_features` returns `value.pooler_output` when present, otherwise returns `value`; `_rows()` only receives the resulting tensor.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_embeddings.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/embeddings/image.py tests/test_embeddings.py
git commit -m "fix: harden CLIP text and feature handling"
```

### Task 3: Optional Vietnamese Chroma Collections and Indexing

**Files:**
- Modify: `src/retrieval/vector_store.py`
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `build_text_document(...)`, `build_clip_text(...)`, `load_jsonl(...)` from Task 1; `ImageEmbedder` behavior from Task 2.
- Produces: required collections `restaurants_text`, `restaurants_image`, `recipes_text`, `recipes_image`; optional collections `dish_text`, `dish_image`, `review_text`; `VectorStore.has_collection(name: str) -> bool`; `index_food_documents(store: VectorStore, records: list[dict | FoodDocument]) -> None`.

- [ ] **Step 1: Write failing backward-compatibility test**

Create a fake old Chroma client that only exposes the original four collections; `VectorStore.open(...)` must succeed and `has_collection("dish_text")` must be false.

- [ ] **Step 2: Write failing optional-search test**

Searching a missing optional `dish_text`/`review_text` collection returns `[]` rather than raising; missing required original collections still raise as before.

- [ ] **Step 3: Write failing indexing tests**

Given one normalized dish and one VLSP review, assert scalar-only metadata, rich SentenceTransformer documents, compact CLIP dish text, and correct collection IDs/documents.

- [ ] **Step 4: Write failing local-image test**

With a small generated PIL image path, assert `dish_image` receives `ImageEmbedder.embed_images(...)` output; a missing image path does not crash and does not fabricate image embeddings.

- [ ] **Step 5: Refactor collection initialization and search dispatch**

Required collections are loaded strictly; optional collections are discovered/created when available. Support `kind in {"restaurants", "recipes", "dish", "review"}` and reject unsupported modality combinations such as `review/image`.

- [ ] **Step 6: Implement `index_food_documents(...)`**

Use dedicated representations, scalar metadata only, and actual image embedding when a valid local dish image exists.

- [ ] **Step 7: Run retrieval tests**

Run: `pytest tests/test_retrieval.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/retrieval/vector_store.py tests/test_retrieval.py
git commit -m "feat: index optional Vietnamese knowledge collections"
```

### Task 4: Explicit Hugging Face Download and Build-Index Integration

**Files:**
- Create: `scripts/download_vietnamese_data.py`
- Modify: `scripts/build_index.py`
- Modify: `requirements.txt`
- Modify: `tests/test_vietnamese_data.py`
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: Task 1 normalizers/JSONL writers; Task 3 `index_food_documents(...)`.
- Produces: CLI `python scripts/download_vietnamese_data.py`; optional files `data/normalized/dishes.jsonl` and `data/normalized/reviews_vi.jsonl`.

- [ ] **Step 1: Write failing downloader tests using a fake dataset loader**

Factor the script around `download_and_normalize(load_dataset_fn=..., output_dir=...)` so tests can verify exact dataset names, normalization, output files, and no network usage.

- [ ] **Step 2: Write failing error-path test**

When `datasets` is unavailable or loading fails, the CLI-facing function must raise/print an actionable dependency/network message and must not fabricate normalized files.

- [ ] **Step 3: Implement downloader with exact dataset identifiers**

Load only the three approved Vietnamese datasets, normalize them through Task 1 functions, and write UTF-8 JSONL under `data/normalized/`.

- [ ] **Step 4: Add `datasets` to `requirements.txt`**

Do not add unrelated dependencies.

- [ ] **Step 5: Write failing `build_index.py` optional-file tests**

Cover both missing normalized files (existing indexing still runs) and present files (dish/review documents are indexed automatically).

- [ ] **Step 6: Modify `scripts/build_index.py`**

Check `data/normalized/dishes.jsonl` and `reviews_vi.jsonl`; if present, load and index them; if absent, continue silently with bundled restaurant/recipe data.

- [ ] **Step 7: Run focused tests**

Run: `pytest tests/test_vietnamese_data.py tests/test_retrieval.py -v`

Expected: PASS without internet.

- [ ] **Step 8: Commit**

```bash
git add scripts/download_vietnamese_data.py scripts/build_index.py requirements.txt tests/test_vietnamese_data.py tests/test_retrieval.py
git commit -m "feat: add offline Vietnamese dataset ingestion"
```

### Task 5: Soft Preference Ranking and Strict Constraint Semantics

**Files:**
- Create: `src/retrieval/ranking.py`
- Modify: `src/retrieval/search.py`
- Modify: `src/retrieval/fusion.py`
- Create: `tests/test_ranking.py`
- Modify: `tests/test_filters.py`
- Modify: `tests/test_fusion.py`

**Interfaces:**
- Consumes: fused candidates with `fusion_score` and `record` metadata.
- Produces: constants `SEMANTIC_WEIGHT = 0.50`, `DISH_WEIGHT = 0.20`, `PREFERENCE_WEIGHT = 0.20`, `REVIEW_WEIGHT = 0.10`; `preference_score(record: dict, preferences: dict, *, dish_terms: list[str] | None = None) -> float`; `rerank_candidates(candidates: list[dict], preferences: dict, *, dish_terms: list[str] | None = None, review_scores: dict[str, float] | None = None) -> list[dict]`; `build_where(preferences: dict | None, *, strict_only: bool = False) -> dict | None`.

- [ ] **Step 1: Write failing ranking tests for semantic relevance plus soft mismatch**

A strong spicy candidate priced `$` must remain in the result set for `budget="moderate"`; an exact `$$` match with similar semantics must outrank it.

- [ ] **Step 2: Write failing dish/preference component tests**

Assert cuisine, dish/menu terms, price, and location contribute according to the spec proportions; missing fields contribute zero rather than causing errors.

- [ ] **Step 3: Write failing strict-filter tests**

`build_where(..., strict_only=True)` must include only explicitly strict constraints; normal soft preferences must not produce Chroma `where` clauses.

- [ ] **Step 4: Run focused tests and confirm failure**

Run: `pytest tests/test_ranking.py tests/test_filters.py tests/test_fusion.py -v`

Expected: FAIL until ranking and new filter semantics exist.

- [ ] **Step 5: Implement transparent normalized ranking**

Normalize retrieval scores before combining; attach `ranking_components` and `ranking_score` to each candidate for inspectability.

- [ ] **Step 6: Refactor filter building to support strict-only mode**

Preserve existing filter-expression generation for callers/tests that explicitly request hard filters, but default runtime retrieval will no longer hard-filter soft preferences.

- [ ] **Step 7: Run focused tests**

Run: `pytest tests/test_ranking.py tests/test_filters.py tests/test_fusion.py -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/retrieval/ranking.py src/retrieval/search.py src/retrieval/fusion.py tests/test_ranking.py tests/test_filters.py tests/test_fusion.py
git commit -m "feat: add soft preference reranking"
```

### Task 6: Runtime Dish Retrieval, Fallback, and Grounded UI Messaging

**Files:**
- Modify: `src/agents/schemas.py`
- Modify: `src/ui/chatbot.py`
- Modify: `tests/test_chatbot.py`
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: Task 3 optional `dish_*` collections and Task 5 `rerank_candidates(...)`.
- Produces: preference extraction that can represent explicit strictness internally; runtime retrieval that searches dish knowledge when available, retrieves restaurant candidates without soft hard-filters, reranks them, and marks relaxed fallback results; `format_workflow_result(...)` distinguishes exact/relaxed/no-match states.

- [ ] **Step 1: Write failing runtime retrieval tests**

Using fake stores/LLM outputs, assert a Vietnamese query searches `dish_text` when available, restaurant search still runs without a soft budget `where`, and missing dish collections fall back to the original restaurant path without error.

- [ ] **Step 2: Write failing fallback tests**

A semantically strong restaurant with a price mismatch must return with `match_type="relaxed"` (or equivalent explicit marker) rather than produce no candidates.

- [ ] **Step 3: Write failing strict no-match test**

An explicitly strict constraint that excludes all candidates must produce a genuine `no_match` and never introduce an unindexed ID.

- [ ] **Step 4: Write failing UI formatting tests**

Relaxed fallback output must state which preference was not satisfied; genuine no-match must say no relevant indexed candidate was found rather than merely suggesting filters were too strict.

- [ ] **Step 5: Implement internal strictness representation**

Add the smallest backward-compatible schema needed to distinguish mandatory constraints from ordinary preferences; existing structured extraction behavior remains valid when strictness is absent.

- [ ] **Step 6: Refactor `build_runtime().retrieve(...)`**

Retrieve dish signals only when optional collections exist, perform text/image restaurant retrieval without soft hard-filters, fuse, rerank, and apply strict constraints only where explicitly required.

- [ ] **Step 7: Add fallback annotations consumed by formatting**

Do not alter candidate IDs or create synthetic restaurants; explanations derive from actual requested-vs-record preference differences.

- [ ] **Step 8: Run runtime/UI tests**

Run: `pytest tests/test_chatbot.py tests/test_retrieval.py tests/test_workflow.py -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/agents/schemas.py src/ui/chatbot.py tests/test_chatbot.py tests/test_retrieval.py
git commit -m "feat: add Vietnamese-aware fallback retrieval"
```

### Task 7: Held-Out Vietnamese Evaluation and Documentation

**Files:**
- Modify: `evaluation/queries.json`
- Modify: `evaluation/evaluate_retrieval.py`
- Modify: `tests/test_evaluation.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: completed retrieval/ranking runtime.
- Produces: a distinct Vietnamese held-out query group and documented setup/limitations; existing evaluation metrics remain Recall@K, Hit Rate@K, and MRR where qrels exist.

- [ ] **Step 1: Write failing evaluation-group test**

Assert Vietnamese held-out queries are separately identifiable from tuning/development examples and cover exact dish, alias, spicy, price fallback, genuine no-match, and dish-assisted restaurant retrieval cases.

- [ ] **Step 2: Update evaluation fixtures/harness without inventing qrels**

Only add qrels for bundled/curated candidates that have defensible expected IDs; no generated recommendation-accuracy claim is added.

- [ ] **Step 3: Update README setup and provenance**

Document:

```bash
pip install -r requirements.txt
python scripts/download_vietnamese_data.py
python scripts/build_index.py
python app.py
```

Also document optional-download behavior, the three HF dataset names, restaurant-directory limitation, ranking weights as provisional, CLIP 77-token handling, and held-out evaluation caveat.

- [ ] **Step 4: Run evaluation tests**

Run: `pytest tests/test_evaluation.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/queries.json evaluation/evaluate_retrieval.py tests/test_evaluation.py README.md
git commit -m "docs: add Vietnamese retrieval evaluation guidance"
```

### Task 8: Full Verification

**Files:**
- Verify all changed files from Tasks 1-7.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: evidence that the feature works offline at unit-test level and preserves existing behavior.

- [ ] **Step 1: Run the complete unit suite**

Run: `pytest -q`

Expected: all tests pass with no network/API requirement.

- [ ] **Step 2: Verify sample-only index behavior**

Run with Vietnamese normalized files absent in a temporary test fixture or equivalent mocked path and confirm `scripts/build_index.py` still indexes restaurant/recipe collections.

- [ ] **Step 3: Verify normalized-file index behavior with fixtures**

Use small local JSONL fixture records, run the index-building path, and confirm `dish_text`, `dish_image` where images exist, and `review_text` receive records.

- [ ] **Step 4: Inspect git diff for provenance and scope**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors, no downloaded HF datasets/images accidentally staged, no secrets, no fabricated restaurant metadata.

- [ ] **Step 5: Final commit if verification required cleanup**

```bash
git add <only-cleanup-files>
git commit -m "test: verify Vietnamese data integration"
```
