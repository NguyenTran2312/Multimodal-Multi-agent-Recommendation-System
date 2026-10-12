# Multimodal Multi-Agent Food Recommendation System

![Architecture](assets/architecture.svg)

## Core idea

Users describe food preferences with a mix of cuisine, budget, atmosphere, dietary constraints, location, and visual expectations. This project separates deterministic retrieval from LLM reasoning:

> The LLM may analyze and rank retrieved candidates, but it cannot invent a restaurant or recipe that was not retrieved.

## Architecture

```text
User query
  ↓
Gradio UI
  ↓
Structured preference extraction
  ↓
Multimodal retrieval
  ├── SentenceTransformer text embeddings
  ├── CLIP text/image space
  ├── Chroma vector search
  ├── metadata filters
  └── score normalization + fusion
  ↓
Retrieved candidates
  ↓
LangGraph specialized agents
  ├── profile
  ├── trend
  ├── style
  └── nutrition
  ↓
Grounded recommendation agent
```

## Tech stack

Python, LangGraph, LangChain, ChromaDB, SentenceTransformers, CLIP, Pydantic, Gradio.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Add your API key to `.env`:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Build the local vector index:

```bash
python scripts/build_index.py
```

Run the app:

```bash
python app.py
```

Or run a CLI smoke test:

```bash
python scripts/smoke_test.py "Find me a cozy Japanese restaurant"
```

## Project structure

```text
src/
├── agents/       # structured specialized agents
├── data/         # loading and offline preprocessing
├── embeddings/   # SentenceTransformer + CLIP wrappers
├── retrieval/    # Chroma search, filters, fusion
├── workflow/     # LangGraph orchestration
└── ui/           # Gradio interface

scripts/          # preprocessing, index build, smoke test
data/             # compact runnable sample data
evaluation/       # retrieval evaluation harness
assets/           # architecture diagram
```

## Engineering highlights

- Reusable text and visual embedding services with lazy model loading.
- Persistent Chroma collections instead of rebuilding the index at startup.
- Structured preference extraction and Pydantic-validated agent outputs.
- Candidate grounding enforced in both the recommender agent and workflow layer.
- Deterministic metadata filtering and multimodal score fusion outside the LLM.
- Controlled errors for missing API keys, missing indexes, empty retrieval results, and provider failures.


## Evaluation discipline

The project includes a held-out retrieval evaluation harness for Precision@K, Recall@K, MRR, and Hit Rate@K. No production-quality accuracy claim is made until the real local index is built and the held-out evaluation is run.

