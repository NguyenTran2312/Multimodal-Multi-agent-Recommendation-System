# Web Search and Dataset Expansion Design

## Goal

Extend the food recommendation system with live external retrieval and additional structured datasets while preserving the existing local RAG workflow and the three familiar portfolio data shapes:

- `restaurants.json`
- `recipes.json`
- `reviews.json`

The system should remain grounded, avoid fabricating unavailable attributes, and clearly distinguish local curated data from live external evidence.

## Scope

This design adds three capabilities:

1. a general-purpose web search tool for fresh culinary information;
2. a structured restaurant provider for real restaurant discovery and metadata;
3. dataset importers that normalize additional recipe/review sources into JSON structures compatible with the existing sample files.

The current Chroma, multimodal retrieval, and agent architecture remain in place. External retrieval augments local retrieval rather than replacing it.

## Recommended external sources

### Tavily for general web search

Use Tavily as the general web-search provider for:

- restaurant websites and menu pages;
- culinary articles and food guides;
- recipe and ingredient information;
- fresh web evidence that is not present in the local index.

The Tavily client should be isolated behind a small project-owned interface so the workflow is not tightly coupled to one provider.

### Google Places for structured restaurant data

Use Google Places as the structured restaurant discovery source when the user asks for real restaurants and the local catalogue is insufficient.

Target fields include, when returned by the provider:

- place/provider ID;
- restaurant name;
- address/location;
- place/cuisine categories;
- price level;
- rating;
- review count/review snippets when available;
- opening-hours information;
- website;
- photos/photo references.

Google Places records are live external entities and must not be permanently rewritten into the bundled sample catalogue by default.

### RecipeNLG for recipe expansion

Use RecipeNLG as an additional recipe source because its title, ingredient, direction, source, and link fields map closely to the existing recipe catalogue.

Normalize selected RecipeNLG rows into a project-compatible recipe JSON representation instead of changing the application's recipe schema to match the source dataset.

### Additional Vietnamese restaurant-review dataset

Add `pqthinh232/vietnamese-restaurant-review-sentiment-dataset` as an additional Vietnamese review source.

Normalize review text, sentiment label, source ID, and language into the project's review representation. Do not create restaurant linkage unless the source actually contains a reliable restaurant/business identifier.

## Retrieval architecture

```text
User query
   ↓
Intent + preference extraction
   ↓
Local retrieval first
   ├── restaurant catalogue
   ├── recipe catalogue
   ├── Vietnamese dish knowledge
   └── review knowledge
   ↓
External retrieval decision
   ├── local evidence sufficient → continue locally
   └── local evidence insufficient or freshness explicitly needed
          ├── Tavily web search
          └── Google Places for real restaurant discovery
   ↓
Normalize external evidence
   ↓
Merge / rerank candidate evidence
   ↓
Grounded final answer with source provenance
```

## External-search decision policy

External retrieval should not run for every request.

Use local retrieval only when:

- a strong local exact match exists;
- the query is clearly about one of the bundled recipe/restaurant candidates;
- freshness is not required.

Use external retrieval when:

- the user explicitly asks for current, nearby, open-now, real-world, recent, or live information;
- the local restaurant catalogue has no strong match;
- the user asks for a cuisine/location not represented in the local catalogue;
- current menus, hours, availability, or recent restaurant information are required;
- the workflow explicitly requests broader external evidence.

The external-search decision should be deterministic where practical and should not silently turn every query into a paid external API call.

## Tool interfaces

### General web search

Add a provider-neutral module such as:

`src/tools/web_search.py`

Proposed interface:

```python
@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str
    published_at: str | None = None

class WebSearchClient:
    def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        ...
```

Provide a Tavily-backed implementation. API credentials must come from environment variables and never be committed.

### Structured restaurant search

Add a separate module such as:

`src/tools/places.py`

Proposed normalized record:

```python
@dataclass(frozen=True)
class PlaceResult:
    provider_id: str
    name: str
    address: str | None
    categories: list[str]
    price_level: str | None
    rating: float | None
    review_count: int | None
    website: str | None
    opening_hours: dict | None
    photo_refs: list[str]
    source: str
```

Keep this separate from Tavily because structured place discovery and open-web search serve different needs.

## External evidence normalization

External data should be converted into project-owned structures before the recommender sees it.

Do not pass provider-specific response objects deep into the workflow.

Every external evidence object should preserve provenance:

- `source`
- `source_url` when available
- `provider_id` or source ID
- retrieval timestamp when appropriate

The final user-facing explanation may cite/source this evidence, but should not expose implementation details such as provider response JSON or API internals.

## Dataset normalization into the three sample-file shapes

### Restaurants

Target compatibility with the existing restaurant catalogue fields:

```json
{
  "id": "...",
  "name": "...",
  "location": "...",
  "cuisine": "...",
  "price": "...",
  "description": "..."
}
```

For structured live sources such as Google Places, populate only fields actually returned or safely derived from provider categories.

Do not infer unsupported cuisine, atmosphere, or menu details from the business name alone.

### Recipes

Target compatibility with the existing recipe catalogue while retaining provenance:

```json
{
  "id": "...",
  "name": "...",
  "cuisine": "...",
  "servings": null,
  "prep_time": null,
  "cook_time": null,
  "total_time": null,
  "ingredients": [],
  "directions": [],
  "image_description": null,
  "source_dataset": "...",
  "source_id": "...",
  "source_url": "..."
}
```

RecipeNLG is expected to provide strong coverage for name/title, ingredients, directions, and source link. Other unsupported target fields stay null/empty.

### Reviews

Target compatibility with the current review file while preserving Vietnamese sentiment metadata:

```json
{
  "reviewId": "...",
  "userId": null,
  "itemId": null,
  "title": null,
  "text": "...",
  "date": null,
  "rating": null,
  "language": "vi",
  "sentiment": "...",
  "images": [],
  "image_captions": [],
  "source_dataset": "...",
  "source_id": "..."
}
```

If a dataset does not identify the reviewed restaurant, `itemId` remains null. Never create synthetic restaurant linkage.

## Data outputs

Keep generated external datasets separate from the original bundled samples.

Recommended files:

```text
data/external/recipes_recipenlg.json
data/external/reviews_vi_sentiment.json
```

Optionally expose a normalized live-cache location for development/testing, but do not commit volatile API responses by default.

Existing bundled files remain unchanged:

```text
data/restaurants.json
data/recipes.json
data/reviews.json
```

The index-building step may load both bundled and normalized external files.

## Configuration

Add environment-backed configuration for external tools, for example:

```text
TAVILY_API_KEY=
GOOGLE_PLACES_API_KEY=
ENABLE_WEB_SEARCH=true
ENABLE_PLACES_SEARCH=true
```

External tools should fail gracefully when credentials are absent. Local RAG must remain usable without either API key.

## Grounding and response behavior

The production prompt policy already requires grounded recommendations. External retrieval adds these additional rules:

- distinguish live external records from local catalogue records;
- never imply a web result was verified beyond the returned evidence;
- do not claim opening hours, price, menu, rating, or availability unless provided by the external source;
- preserve URLs/provenance internally for citation or traceability;
- when local and external candidates are mixed, prefer the candidate that best satisfies the user, not automatically the external result;
- avoid duplicate entities returned by multiple sources when a stable identity can be determined.

## Testing strategy

All provider integrations should be unit-tested with fake clients or mocked HTTP/provider responses. Normal test runs must not call paid/live APIs.

Tests should cover:

1. Tavily response normalization into `WebSearchResult`;
2. Google Places response normalization into `PlaceResult`;
3. missing API keys preserve local-only behavior;
4. external-search decision does not trigger on strong local matches;
5. external-search decision triggers for explicit fresh/current/real-world requests;
6. RecipeNLG rows normalize without invented fields;
7. Vietnamese review rows normalize with null `itemId` when restaurant identity is absent;
8. provenance fields survive normalization;
9. external candidate claims remain grounded;
10. duplicate external/local restaurant candidates can be deduplicated by stable identity when available.

## Files expected to change

Likely additions/modifications:

```text
src/tools/__init__.py
src/tools/web_search.py
src/tools/places.py
src/config.py
src/workflow/recommendation.py
src/ui/chatbot.py
src/data/external.py
scripts/download_external_data.py
scripts/build_index.py
requirements.txt
.env.example
tests/test_web_search.py
tests/test_places.py
tests/test_external_data.py
tests/test_chatbot.py
```

The exact implementation plan may reduce this list if the existing data/preprocessing modules already provide reusable normalization hooks.

## Non-goals

This phase does not:

- guarantee reservation booking;
- guarantee live table availability;
- scrape arbitrary websites directly;
- permanently merge volatile live restaurant records into the hand-curated sample catalogue;
- fabricate missing source attributes;
- replace local Chroma retrieval with web search;
- add a general non-food browsing assistant.

## Success criteria

The feature is successful when:

- the chatbot can optionally use live web search for culinary information;
- real restaurant discovery can use structured place data instead of fictional inference;
- local-only operation still works with no external API keys;
- RecipeNLG-derived records fit the project recipe schema without fabricated fields;
- Vietnamese review records fit the project review schema without fabricated restaurant linkage;
- external records retain source provenance;
- tests remain deterministic and make no paid/live API calls;
- final answers stay concise, grounded, and culinary-focused.
