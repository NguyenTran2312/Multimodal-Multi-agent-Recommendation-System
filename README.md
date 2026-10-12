# Multimodal Multi-Agent Food Recommendation System

A portfolio-grade AI recommendation application refactored from a sequence of course labs into one runnable Python project. The system combines semantic retrieval, CLIP-compatible visual retrieval, metadata filtering, multimodal score fusion, structured LLM agents, LangGraph orchestration, and a Gradio chat interface.

![Architecture](assets/architecture.svg)

## Problem

Food requests are usually underspecified and subjective: a user may ask for "a cozy inexpensive Japanese place" or "something colorful with tofu I can cook at home." A useful recommender needs to combine semantic meaning, structured constraints, visual/style cues, and explanation without allowing the LLM to invent options that are absent from the data.

This project therefore follows one core rule:

> The LLM may reason over retrieved candidates, but the final recommendation must reference a candidate that actually came from retrieval.

## What the system does

- interprets restaurant vs. recipe intent;
- extracts structured cuisine, budget, atmosphere, dietary, and location preferences;
- retrieves candidates from a persistent Chroma index;
- uses SentenceTransformers for semantic text embeddings;
- exposes CLIP image and text encoders for a shared visual-semantic space;
- applies metadata constraints where structured metadata is available;
- normalizes modality scores and fuses text/visual rankings;
- runs profile, trend, style, and nutrition analysis agents;
- orchestrates the agent branches with LangGraph;
- validates structured LLM outputs with Pydantic;
- rejects final recommendations that reference non-retrieved IDs;
- serves the result through a Gradio chat UI.

## Architecture

```text
User
  ↓
Gradio
  ↓
Intent + structured preference extraction
  ↓
Local retrieval
  ├── SentenceTransformer text embeddings
  ├── CLIP joint-space embeddings
  ├── Chroma vector search
  ├── metadata filtering
  └── normalized score fusion
  ↓
Retrieved candidates
  ↓
LangGraph
  ├── profile analysis
  ├── trend analysis
  ├── style analysis
  └── nutrition analysis
  ↓
Grounded recommendation agent
  ↓
Restaurant / recipe response
```

The expensive API-backed model is used for language reasoning. Data loading, embeddings, vector retrieval, filtering, and fusion remain local.

## Repository structure

```text
.
├── app.py
├── src/
│   ├── agents/          # structured specialized agents
│   ├── data/            # loading + offline preprocessing
│   ├── embeddings/      # reusable SentenceTransformer / CLIP wrappers
│   ├── retrieval/       # Chroma search + metadata filters + fusion
│   ├── ui/              # Gradio adapter
│   └── workflow/        # LangGraph workflow
├── scripts/
│   ├── preprocess_data.py
│   ├── build_index.py
│   └── smoke_test.py
├── evaluation/
│   ├── queries.json
│   └── evaluate_retrieval.py
├── tests/
├── data/
├── assets/
└── notebooks/original_labs/
```

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

Then set:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Never commit `.env`.

## Data preparation

The repository already contains normalized portfolio datasets. To rerun normalization:

```bash
python scripts/preprocess_data.py \
  --restaurants data/restaurants.json \
  --recipes data/recipes.json \
  --reviews data/reviews.json \
  --output-dir data/processed
```

Image-caption generation is deliberately not performed at application startup. Existing captions/descriptions are treated as offline data assets.

## Build the local vector index

```bash
python scripts/build_index.py
```

The first run may download the configured SentenceTransformer and CLIP models. The index is persisted under `data/chroma/` and is not rebuilt when the application starts.

### Visual-channel note

`ImageEmbedder` supports raw image embedding with CLIP. The bundled portfolio data does not include all original image binaries, so `build_index.py` uses the supplied visual descriptions/captions through CLIP's text encoder for the persisted visual channel. The preserved original lab notebooks show the raw-image indexing exercise used during the course.

## Run the app

```bash
python app.py
```

Example prompts:

```text
I want an inexpensive cozy Japanese restaurant.

Find me a colorful tofu recipe I can make at home.
```

You can also exercise the full runtime without launching Gradio:

```bash
python scripts/smoke_test.py "Find me a cozy Japanese restaurant"
```

Missing API keys or an unbuilt index return actionable errors rather than opaque provider exceptions.

## Testing

```bash
pytest -v
```

The test suite covers configuration, malformed data, real uploaded review-image formats, embedding wrapper reuse, metadata filters, missing indexes, retrieval, score fusion, structured agent output, candidate grounding, workflow state propagation, no-result behavior, provider failures, direct CLI execution, and retrieval metrics.

Unit tests use fake model backends and do not download models or call paid APIs.

## Evaluation

`evaluation/queries.json` separates `dev` and `heldout` query records. The held-out records are not used to choose fusion weights.

Run:

```bash
python evaluation/evaluate_retrieval.py
```

The report calculates:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Hit Rate@K

for text-only, visual-only, and fused retrieval.

### Current results

No retrieval-quality number is claimed in this repository until a real local index has been built and the held-out runner has been executed. The evaluation command intentionally refuses to fabricate metrics when the index is missing.

This small fictional dataset is suitable for demonstrating retrieval mechanics and controlled comparisons; it does not establish performance on real-world restaurant or recipe traffic.

## Error handling

The application explicitly handles:

- missing API keys;
- missing persisted vector indexes;
- malformed JSON files;
- invalid/broken image references during preprocessing;
- metadata filters that return no candidates;
- provider failures;
- recommendations containing IDs that were not retrieved.

## From labs to application

The original notebooks are preserved unchanged under `notebooks/original_labs/`, but they are not required for normal execution.

| Lab area | Portfolio module |
|---|---|
| multimodal preprocessing | `src/data/` |
| vector index construction | `src/embeddings/`, `src/retrieval/vector_store.py` |
| similarity + metadata filtering | `src/retrieval/search.py` |
| multimodal score fusion | `src/retrieval/fusion.py` |
| specialized agents | `src/agents/` |
| multi-agent system | `src/workflow/` |
| chatbot interface | `src/ui/`, `app.py` |

## Engineering choices

**Why local retrieval?** Embeddings, filtering, and ranking are deterministic operations that do not require an LLM call.

**Why structured agent outputs?** Pydantic schemas convert prompt responses into explicit application contracts and reject malformed output.

**Why candidate grounding?** A generative model can explain and rank retrieved candidates, but it cannot insert an unseen restaurant or recipe into the final result.

**Why separate preprocessing?** Expensive enrichment belongs in a reproducible offline step, not in every application startup.

## Limitations

- The datasets are fictional/course-provided and small.
- The bundled visual channel relies on supplied visual descriptions when original image binaries are unavailable.
- The restaurant JSON is reconstructed from the supplied culinary-map text because the lab-referenced structured restaurant file was not included in the uploaded files.
- The current held-out set is intentionally small and should not be presented as evidence of production recommendation quality.
- External image URLs may expire.

## CV-ready summary

**Multimodal Multi-Agent Food Recommendation System — Python, LangGraph, ChromaDB, SentenceTransformers, CLIP, Pydantic, Gradio**

- Refactored seven instructional AI labs into a modular end-to-end restaurant and recipe recommendation application.
- Built deterministic semantic retrieval with metadata filtering and multimodal score normalization/fusion over persistent Chroma collections.
- Designed structured profile, trend, style, nutrition, and recommendation agents orchestrated with LangGraph and constrained to retrieved candidate IDs.
- Added offline preprocessing, reproducible CLI workflows, automated tests, and a held-out retrieval-evaluation harness while avoiding unsupported production-quality claims.

## Optional Vietnamese knowledge layer

The app can be enriched with Vietnamese-only Hugging Face data without downloading anything at application startup. The approved sources are:

- `HoagMin/Vietnamese_StreetFood_Dataset` for Vietnamese dish images/classes;
- `Tamir39/foodlensvn` for Vietnamese food images and QA/attribute signals;
- `phuongnt101/VLSP2018-ABSA-Restaurant` for Vietnamese restaurant-review text and aspect sentiment.

Download and normalize them explicitly, then rebuild the local index:

```bash
pip install -r requirements.txt
python scripts/download_vietnamese_data.py
python scripts/build_index.py
python app.py
```

Normalized files are written under `data/normalized/`. If those files are absent, the original restaurant/recipe indexing path still works. The downloader never fabricates restaurant names, locations, prices, ratings, or links between VLSP reviews and restaurants.

Vietnamese dish records are indexed in optional `dish_text` and `dish_image` collections; review records go to `review_text`. Restaurant recommendations still come from the restaurant catalogue, while dish retrieval supplies Vietnamese semantic/visual knowledge. Budget, cuisine, and location are soft ranking preferences by default, with explicit `strict_constraints` reserved for genuine must-have filters.

The Street Food source is distributed as a ZIP/YOLO-style archive, so the downloader extracts its image/label layout directly. FoodLens images are materialized locally when the dataset yields image objects. No Hugging Face download occurs when `app.py` starts.

### Vietnamese evaluation

`evaluation/queries.json` includes a separate `heldout_vi` group for Vietnamese prompts. Gold IDs are intentionally left empty until the restaurant catalogue contains validated Vietnamese targets; the evaluator skips unlabelled cases rather than inventing qrels. Do not claim Vietnamese retrieval quality until that group has manually validated labels and has been run against a real built index.
