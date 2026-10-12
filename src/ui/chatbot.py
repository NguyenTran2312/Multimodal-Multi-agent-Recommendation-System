from __future__ import annotations

import os

from src.agents._common import invoke_structured
from src.agents.nutrition import analyze_nutrition
from src.agents.profile import build_profile
from src.agents.recommender import generate_recommendations
from src.agents.schemas import UserPreferences
from src.agents.style import analyze_style
from src.agents.trends import analyze_trends
from src.config import Settings
from src.retrieval.fusion import fuse_rank
from src.retrieval.ranking import rerank_candidates
from src.retrieval.vector_store import VectorStore
from src.workflow.recommendation import (
    CompiledGraphWorkflow,
    RecommendationWorkflow,
    WorkflowError,
    build_recommendation_graph,
)


PREFERENCE_PROMPT = (
    "Extract explicit food preferences. Use empty lists/null for unspecified fields; "
    "do not invent constraints. Put a preference in strict_constraints only when the user "
    "clearly makes it mandatory (for example: must, only, exactly, under a hard limit)."
)


def classify_intent(query: str) -> str:
    lowered = query.lower()
    if any(token in lowered for token in ("recipe", "cook", "make at home", "ingredients")):
        return "recipe"
    return "restaurant"


def extract_preferences(query: str, llm) -> UserPreferences:
    return invoke_structured(llm, UserPreferences, PREFERENCE_PROMPT, {"query": query})


def _strict_dict(preferences: UserPreferences) -> dict:
    strict = preferences.strict_constraints
    if hasattr(strict, "model_dump"):
        return strict.model_dump(exclude_none=True)
    return dict(strict or {})


def retrieve_candidates(store, query: str, preferences: UserPreferences, *, candidate_depth: int = 20, retrieval_k: int = 5) -> list[dict]:
    kind = "recipes" if classify_intent(query) == "recipe" else "restaurants"
    all_preferences = preferences.model_dump(exclude_none=True)
    soft_prefs = {key: value for key, value in all_preferences.items() if key in {"cuisine", "budget", "price", "location", "atmosphere"} and value}
    strict = _strict_dict(preferences)

    dish_terms: list[str] = []
    if kind == "restaurants" and store.has_collection("dish_text"):
        for item in store.search("dish", query, {}, k=min(candidate_depth, 5), modality="text"):
            record = item.get("record") or {}
            term = record.get("name_vi") or record.get("name")
            if term:
                dish_terms.append(str(term))
            elif item.get("document"):
                dish_terms.append(str(item["document"]).split("|")[0].strip())

    search_preferences = {"strict_constraints": strict} if strict else {}
    text_results = store.search(kind, query, search_preferences, k=candidate_depth, modality="text", strict_only=bool(strict))
    image_results = store.search(kind, query, search_preferences, k=candidate_depth, modality="image", strict_only=bool(strict))
    fused = fuse_rank(text_results, image_results)
    return rerank_candidates(fused, soft_prefs, dish_terms=dish_terms)[:retrieval_k]


def build_runtime() -> RecommendationWorkflow:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    settings = Settings.from_env()
    store = VectorStore.open(settings.chroma_path)
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "langchain-openai is required. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    llm = ChatOpenAI(
        api_key=settings.api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0,
    )

    def retrieve(query: str, preferences: UserPreferences) -> list[dict]:
        return retrieve_candidates(
            store, query, preferences,
            candidate_depth=settings.candidate_depth,
            retrieval_k=settings.retrieval_k,
        )

    workflow = RecommendationWorkflow(
        preference_extractor=lambda query: extract_preferences(query, llm),
        retrieve=retrieve,
        profile_agent=lambda preferences: build_profile(preferences, llm),
        trend_agent=lambda candidates, preferences: analyze_trends(candidates, preferences, llm),
        style_agent=lambda candidates, preferences: analyze_style(candidates, preferences, llm),
        nutrition_agent=lambda candidates, preferences: analyze_nutrition(candidates, preferences, llm),
        recommender=lambda candidates, analyses, preferences: generate_recommendations(
            candidates, analyses, preferences, llm
        ),
    )
    return CompiledGraphWorkflow(build_recommendation_graph(workflow))


def format_workflow_result(result: dict) -> str:
    if result.get("status") == "no_match":
        preferences = result.get("preferences")
        strict = _strict_dict(preferences) if isinstance(preferences, UserPreferences) else {}
        if strict:
            return "I couldn't find a relevant candidate in the indexed data that satisfies the mandatory constraints in your request."
        return "I couldn't find a relevant restaurant or recipe in the indexed data for this request."
    candidates = {str(item["id"]): item for item in result.get("candidates", [])}
    response = result["recommendations"]
    lines = []
    for recommendation in sorted(response.recommendations, key=lambda item: item.rank):
        candidate = candidates.get(recommendation.candidate_id)
        if candidate is None:
            raise ValueError(
                f"Recommendation {recommendation.candidate_id!r} is not present in workflow candidates"
            )
        record = candidate.get("record", {})
        name = record.get("name") or record.get("title") or recommendation.candidate_id
        note = ""
        if candidate.get("match_type") == "relaxed":
            missed = ", ".join(candidate.get("preference_mismatches", []))
            note = f" (closest match; differs on {missed})" if missed else " (closest match)"
        lines.append(f"{recommendation.rank}. **{name}** — {recommendation.reason}{note}")
    return "\n\n".join(lines) if lines else "No grounded recommendation was produced."


def handle_message(message: str, workflow: RecommendationWorkflow) -> str:
    try:
        return format_workflow_result(workflow.recommend(message))
    except WorkflowError as exc:
        return f"I could not complete the recommendation: {exc}"


def create_app(workflow: RecommendationWorkflow | None = None):
    import gradio as gr

    runtime = workflow or build_runtime()

    def chat(message, history):
        del history
        return handle_message(message, runtime)

    with gr.Blocks(title="Multimodal Food Recommender") as app:
        gr.Markdown("# Multimodal Food Recommender")
        gr.Markdown("Ask for a restaurant or recipe using cuisine, budget, atmosphere, location, or dietary preferences.")
        gr.ChatInterface(fn=chat)
    return app