# Web Search and Dataset Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional live culinary web/restaurant retrieval and normalize additional recipe/review datasets into structures compatible with the project's existing restaurant, recipe, and review data shapes.

**Architecture:** Keep local Chroma retrieval as the default path. Add provider-neutral Tavily and Google Places adapters, deterministic external-search routing, and dataset-specific normalization utilities; external evidence is normalized into project-owned records before it reaches recommendation logic. Local-only operation must remain functional when external API keys are missing.

**Tech Stack:** Python 3.11+, dataclasses, `requests` or provider SDKs, Hugging Face `datasets`, ChromaDB, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-web-search-and-dataset-expansion-design.md`

## Global Constraints

- Local retrieval remains the default and must work without Tavily or Google Places credentials.
- External search must not run on every request; it is triggered only by explicit freshness/real-world needs or insufficient local evidence.
- Never fabricate missing restaurant, recipe, or review fields.
- Do not synthesize restaurant linkage for review datasets that lack a reliable restaurant/business identifier.
- Preserve provenance (`source`, `source_url`, `source_dataset`, source/provider IDs, and retrieval timestamp where relevant).
- Do not permanently merge volatile live restaurant API responses into the bundled curated sample files.
- Unit tests must use fake clients or mocked responses and must not call paid/live APIs.
- Existing bundled `data/restaurants.json`, `data/recipes.json`, and `data/reviews.json` remain valid and unchanged.
- Existing Chroma, embedding models, and ranking weights are not replaced by this work.

## Review Focus

- Missing or blank external API keys must produce local-only behavior rather than startup failure; covered in Task 1 and Task 4.
- Provider responses with missing optional fields must normalize without KeyError or invented values; covered in Tasks 2 and 3.
- A query containing words like “current”, “nearby”, “open now”, or a location absent from the local catalogue should trigger external retrieval, while a strong local exact match should not; covered in Task 4.
- External and local restaurant results referring to the same stable provider/business identity should not produce duplicate final candidates; covered in Task 4.
- Dataset rows with absent restaurant linkage, cuisine, timing, rating, or image data must preserve null/empty values instead of heuristic fabrication; covered in Task 5.

---

### Task 1: External Tool Configuration

**Files:**
- Modify: `src/config.py`
- Modify: `.env.example`
- Modify: `requirements.txt`
- Create or modify: `tests/test_config.py`

**Interfaces:**
- Consumes: existing `Settings.from_env()` and local OpenAI configuration.
- Produces: `Settings.tavily_api_key: str | None`, `Settings.google_places_api_key: str | None`, `Settings.enable_web_search: bool`, `Settings.enable_places_search: bool`.

- [ ] **Step 1: Write failing configuration tests**

Add tests asserting that:
- blank/missing `TAVILY_API_KEY` and `GOOGLE_PLACES_API_KEY` become `None`;
- `ENABLE_WEB_SEARCH` and `ENABLE_PLACES_SEARCH` default to false unless explicitly enabled;
- missing external keys never cause `Settings.from_env()` to fail when `OPENAI_API_KEY` exists.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/test_config.py -v`

Expected: FAIL because the external settings do not exist yet.

- [ ] **Step 3: Extend `Settings` and environment parsing**

Implement the exact fields above in `src/config.py`. Parse enable flags from common truthy values (`1`, `true`, `yes`, `on`) case-insensitively. Keep the existing OpenAI-key requirement unchanged.

- [ ] **Step 4: Update `.env.example` and dependencies**

Add:

```text
TAVILY_API_KEY=
GOOGLE_PLACES_API_KEY=
ENABLE_WEB_SEARCH=false
ENABLE_PLACES_SEARCH=false
```

Add only the minimal client dependency needed by the chosen implementation; prefer plain `requests` if it avoids unnecessary SDK coupling.

- [ ] **Step 5: Run focused tests and verify pass**

Run: `pytest tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/config.py .env.example requirements.txt tests/test_config.py
git commit -m "feat: configure external culinary search providers"
```

### Task 2: Tavily Web Search Adapter

**Files:**
- Create: `src/tools/__init__.py`
- Create: `src/tools/web_search.py`
- Create: `tests/test_web_search.py`

**Interfaces:**
- Consumes: `api_key: str`, an injectable HTTP/client callable for tests.
- Produces:
  - `WebSearchResult(title: str, url: str, snippet: str, source: str, published_at: str | None = None)`
  - `TavilyWebSearchClient.search(query: str, *, max_results: int = 5) -> list[WebSearchResult]`

- [ ] **Step 1: Write failing response-normalization tests**

Test that a fake Tavily response:
- maps title, URL, content/snippet, source=`"tavily"`, and optional published date;
- skips malformed entries that have neither URL nor useful text;
- caps results at `max_results`;
- preserves empty optional values as `None`/empty strings rather than inventing content.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/test_web_search.py -v`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement the provider-neutral result type and Tavily client**

Use an injectable transport/client so tests never perform network calls. Raise a clear configuration error only when the Tavily client is explicitly constructed without an API key; do not affect local-only startup.

- [ ] **Step 4: Run focused tests and verify pass**

Run: `pytest tests/test_web_search.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/__init__.py src/tools/web_search.py tests/test_web_search.py
git commit -m "feat: add Tavily culinary web search adapter"
```

### Task 3: Google Places Restaurant Adapter

**Files:**
- Create: `src/tools/places.py`
- Create: `tests/test_places.py`

**Interfaces:**
- Consumes: `api_key: str`, an injectable HTTP/client callable for tests.
- Produces:
  - `PlaceResult(provider_id: str, name: str, address: str | None, categories: list[str], price_level: str | None, rating: float | None, review_count: int | None, website: str | None, opening_hours: dict | None, photo_refs: list[str], source: str)`
  - `GooglePlacesClient.search(query: str, *, max_results: int = 5) -> list[PlaceResult]`

- [ ] **Step 1: Write failing normalization tests**

Test complete and sparse provider responses. Assert that:
- provider/business identity is preserved;
- absent address, price, rating, website, hours, or photos remain `None`/empty;
- categories are copied from provider types only;
- no cuisine or atmosphere is inferred from the business name;
- result count respects `max_results`.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/test_places.py -v`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement `PlaceResult` and `GooglePlacesClient`**

Use the current Google Places text-search endpoint and request only the fields needed by the normalized model. Keep provider response handling inside this module.

- [ ] **Step 4: Run focused tests and verify pass**

Run: `pytest tests/test_places.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/places.py tests/test_places.py
git commit -m "feat: add structured restaurant place search"
```

### Task 4: External Search Routing and Candidate Normalization

**Files:**
- Create: `src/retrieval/external.py`
- Modify: `src/ui/chatbot.py`
- Modify: `src/workflow/recommendation.py` only if query/evidence plumbing requires it
- Modify: `tests/test_chatbot.py`
- Create or modify: `tests/test_external_retrieval.py`

**Interfaces:**
- Consumes: local ranked candidates, `UserPreferences`, original query, optional `WebSearchClient`, optional `GooglePlacesClient`.
- Produces:
  - `should_use_external_search(query: str, local_candidates: list[dict], preferences: UserPreferences) -> bool`
  - `place_to_candidate(place: PlaceResult) -> dict`
  - `web_result_to_evidence(result: WebSearchResult) -> dict`
  - `deduplicate_candidates(candidates: list[dict]) -> list[dict]`

- [ ] **Step 1: Write failing routing tests**

Add cases asserting:
- strong exact local matches do not trigger external search;
- explicit freshness/real-world terms (`current`, `nearby`, `open now`, `today`, `real restaurant`, Vietnamese equivalents where practical) do trigger it;
- no/weak local restaurant match triggers it when an external provider is enabled;
- disabled providers or absent keys preserve local-only behavior.

- [ ] **Step 2: Write failing normalization/deduplication tests**

Assert that:
- a `PlaceResult` becomes a restaurant-shaped candidate with provenance and no fabricated cuisine/menu data;
- a `WebSearchResult` becomes evidence, not a fake restaurant candidate;
- duplicate place candidates with the same provider ID collapse to one record;
- local/external records only deduplicate when a stable identity match exists.

- [ ] **Step 3: Run focused tests and verify failure**

Run: `pytest tests/test_external_retrieval.py tests/test_chatbot.py -k "external or web or place" -v`

Expected: FAIL because routing and adapters are not wired in.

- [ ] **Step 4: Implement deterministic external routing in `src/retrieval/external.py`**

Keep the policy small and explicit. External search is supplemental; it must not replace the existing `retrieve_candidates()` local path.

- [ ] **Step 5: Wire optional provider clients in `build_runtime()`**

Construct Tavily/Places clients only when the corresponding enable flag is true and a key exists. Pass them into retrieval through closures or a small runtime dependency object; do not introduce module-level network clients.

- [ ] **Step 6: Merge normalized external candidates/evidence after local retrieval**

For restaurant intent, Places may contribute real restaurant candidates. Tavily contributes evidence/provenance only unless a later explicit parser validates a business entity. Preserve the existing maximum final candidate count and grounding rules.

- [ ] **Step 7: Run focused tests and verify pass**

Run: `pytest tests/test_external_retrieval.py tests/test_chatbot.py -k "external or web or place" -v`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/retrieval/external.py src/ui/chatbot.py src/workflow/recommendation.py tests/test_external_retrieval.py tests/test_chatbot.py
git commit -m "feat: add optional external culinary retrieval"
```

### Task 5: RecipeNLG and Vietnamese Review Normalization

**Files:**
- Create: `src/data/external.py`
- Create: `tests/test_external_data.py`

**Interfaces:**
- Consumes: raw RecipeNLG rows and `pqthinh232/vietnamese-restaurant-review-sentiment-dataset` rows.
- Produces:
  - `normalize_recipenlg_row(row: dict, *, source_id: str) -> dict`
  - `normalize_vi_review_row(row: dict, *, source_id: str, split: str | None = None) -> dict`

- [ ] **Step 1: Write failing RecipeNLG normalization tests**

Assert mapping for title/name, ingredients, directions, source URL/provenance, and stable source ID. Assert unsupported cuisine, timings, servings, and image description remain null/empty rather than guessed.

- [ ] **Step 2: Write failing Vietnamese review normalization tests**

Assert review text, sentiment, language=`"vi"`, source dataset, source ID, and optional split are preserved. Assert `itemId`, user ID, rating, title, date, and images remain null/empty when absent.

- [ ] **Step 3: Run focused tests and verify failure**

Run: `pytest tests/test_external_data.py -v`

Expected: FAIL because the external normalizers do not exist.

- [ ] **Step 4: Implement `src/data/external.py`**

Use deterministic IDs derived from source IDs. Match the current sample-file field names where possible and include provenance fields as additive metadata.

- [ ] **Step 5: Run focused tests and verify pass**

Run: `pytest tests/test_external_data.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/data/external.py tests/test_external_data.py
git commit -m "feat: normalize external recipe and review datasets"
```

### Task 6: External Dataset Download Script

**Files:**
- Create: `scripts/download_external_data.py`
- Modify: `.gitignore` if generated files are not already excluded
- Modify: `tests/test_external_data.py`

**Interfaces:**
- Consumes: Hugging Face `datasets.load_dataset`, Task 5 normalizers.
- Produces:
  - `data/external/recipes_recipenlg.json`
  - `data/external/reviews_vi_sentiment.json`

- [ ] **Step 1: Write failing script-level tests with fake dataset loaders**

Assert that the script:
- creates `data/external/`;
- writes deterministic UTF-8 JSON arrays;
- supports a row limit for development/tests;
- does not require Tavily, Google Places, or OpenAI credentials;
- never writes live API restaurant responses into committed sample files.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/test_external_data.py -k download -v`

Expected: FAIL because the downloader does not exist.

- [ ] **Step 3: Implement `scripts/download_external_data.py`**

Add CLI options such as `--recipe-limit` and `--review-limit` with `0`/omitted meaning provider default/full configured download. Keep dataset names as constants in `src/data/external.py` or the script.

- [ ] **Step 4: Run focused tests and verify pass**

Run: `pytest tests/test_external_data.py -k download -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/download_external_data.py src/data/external.py tests/test_external_data.py .gitignore
git commit -m "feat: add external culinary dataset downloader"
```

### Task 7: Index External Normalized Datasets

**Files:**
- Modify: `scripts/build_index.py`
- Modify: `src/data/loader.py` or add focused loaders in `src/data/external.py` if necessary
- Modify: `tests/test_retrieval.py` or `tests/test_external_data.py`

**Interfaces:**
- Consumes: bundled restaurant/recipe files plus optional `data/external/recipes_recipenlg.json` and `data/external/reviews_vi_sentiment.json`.
- Produces: indexed recipe and review records without requiring those external files to exist.

- [ ] **Step 1: Write failing optional-index tests**

Assert that:
- index building works when external files are absent;
- RecipeNLG-derived recipe records are added to the recipe collection when the file exists;
- Vietnamese external reviews are added to review text knowledge with provenance;
- no duplicate source IDs are inserted within one external file.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/test_external_data.py tests/test_retrieval.py -k "external or index" -v`

Expected: FAIL because `build_index.py` ignores the new files.

- [ ] **Step 3: Extend index loading**

Load the optional external JSON files only when present. Reuse current `index_records()` / `index_food_documents()` paths where their expected shapes fit; otherwise add a narrow adapter instead of changing vector-store semantics.

- [ ] **Step 4: Run focused tests and verify pass**

Run: `pytest tests/test_external_data.py tests/test_retrieval.py -k "external or index" -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_index.py src/data/loader.py src/data/external.py tests/test_external_data.py tests/test_retrieval.py
git commit -m "feat: index normalized external culinary data"
```

### Task 8: Grounding, Prompt, and Full Regression Verification

**Files:**
- Modify only if required by a failing regression: `src/agents/prompt_policy.py`, `src/agents/recommender.py`, related tests.
- Test: full `tests/` suite.

**Interfaces:**
- Consumes: external provenance/evidence from Tasks 2-7.
- Produces: verified user-facing behavior that distinguishes grounded external evidence from local data without leaking provider internals.

- [ ] **Step 1: Add/adjust grounding tests for external evidence**

Assert that final prompt contracts:
- allow source-backed live facts;
- forbid inventing hours, menu items, price, ratings, or availability when absent;
- do not expose raw provider JSON or API implementation details;
- preserve source URLs/provenance for traceability.

- [ ] **Step 2: Run prompt and external focused tests**

Run: `pytest tests/test_prompt_policy.py tests/test_agent_prompts.py tests/test_external_retrieval.py tests/test_web_search.py tests/test_places.py tests/test_external_data.py -v`

Expected: PASS.

- [ ] **Step 3: Run the complete test suite**

Run: `pytest -v`

Expected: all tests PASS. Report any unrelated pre-existing failures explicitly.

- [ ] **Step 4: Run syntax/whitespace verification**

Run:

```bash
python -m compileall src scripts
git diff --check
```

Expected: compile succeeds and `git diff --check` prints no errors.

- [ ] **Step 5: Verify local-only startup contract**

With `ENABLE_WEB_SEARCH=false` and `ENABLE_PLACES_SEARCH=false`, instantiate runtime/config and run existing smoke tests without external keys.

Expected: existing local recommendation path remains operational.

- [ ] **Step 6: Commit any verification-driven fixes**

```bash
git add src scripts tests .env.example requirements.txt
git commit -m "test: verify external culinary retrieval integration"
```
