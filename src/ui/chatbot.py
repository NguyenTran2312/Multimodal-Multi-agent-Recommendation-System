from __future__ import annotations

import os

from pydantic import BaseModel, Field

from src.agents._common import invoke_structured
from src.agents.nutrition import analyze_nutrition
from src.agents.profile import build_profile
from src.agents.prompt_policy import compose_prompt
from src.agents.recommender import generate_recommendations
from src.agents.schemas import UserPreferences
from src.agents.style import analyze_style
from src.agents.trends import analyze_trends
from src.config import Settings
from src.data.vietnamese import FOODLENS_SOURCE, STREETFOOD_SOURCE
from src.retrieval.external import (
    deduplicate_candidates,
    restaurant_evidence_to_candidate,
    should_use_external_search,
    web_result_to_evidence,
)
from src.retrieval.fusion import fuse_rank
from src.retrieval.ranking import rerank_candidates
from src.retrieval.vector_store import VectorStore
from src.tools.web_search import RestaurantWebEvidence, TavilyWebSearchClient
from src.workflow.recommendation import (
    CompiledGraphWorkflow,
    RecommendationWorkflow,
    WorkflowError,
    build_recommendation_graph,
)


PREFERENCE_PROMPT = compose_prompt(
    """
TASK: Extract only explicit culinary preferences into the UserPreferences schema.

Interpret English and Vietnamese culinary wording, including established cuisine and dish
names. Preserve cuisine, budget, atmosphere, dietary preference, and location intent when
stated. Treat ordinary preference language such as "prefer", "would like", "ưu tiên", or
"muốn" as soft preferences. Put a value in strict_constraints only when the user clearly
makes it mandatory with wording such as "must", "only", "exactly", a hard upper/lower
limit, "chỉ", "bắt buộc", or an equivalent unambiguous requirement.

Do not infer an unstated location, budget, allergy, dietary status, cuisine, atmosphere, or
other constraint. Normalize obvious synonyms only when their meaning is unambiguous. Use
empty lists or null for fields the user did not specify. Do not turn a soft preference into
a hard filter.
"""
)

WEB_RESTAURANT_PROMPT = compose_prompt(
    """
TASK: Convert Tavily web-search results into grounded restaurant evidence.

Return a restaurant only when the supplied title, snippet, or URL gives enough evidence
that a specific restaurant is being described. Extract only fields explicitly supported by
the supplied result. Never infer cuisine, location, price, rating, menu, or opening hours
from the restaurant name alone. Preserve the exact source URL. If a field is unsupported,
leave it null or empty. Omit generic articles, list pages, and results that do not identify
a specific restaurant.
"""
)


class RestaurantWebEvidenceBatch(BaseModel):
    restaurants: list[RestaurantWebEvidence] = Field(default_factory=list)


def classify_intent(query: str) -> str:
    lowered = query.casefold()
    recipe_terms = ("recipe", "cook", "make at home", "ingredients", "công thức", "nấu")
    restaurant_terms = (
        "restaurant", "restaurants", "quán", "nhà hàng", "dinner", "open now",
        "nearby", "địa chỉ", "district", "compare two", "restaurant option",
    )
    dish_terms = (
        "dish", "meal", "food", "món", "something similar", "alternative",
        "noodles", "noodle", "bún", "phở", "pho", "bánh", "cơm", "rice",
    )
    if any(token in lowered for token in recipe_terms):
        return "recipe"
    if any(token in lowered for token in restaurant_terms):
        return "restaurant"
    if any(token in lowered for token in dish_terms):
        return "dish"
    return "restaurant"


def extract_preferences(query: str, llm) -> UserPreferences:
    return invoke_structured(llm, UserPreferences, PREFERENCE_PROMPT, {"query": query})


def extract_restaurant_web_candidates(results, query: str, llm) -> list[RestaurantWebEvidence]:
    payload = {
        "query": query,
        "results": [item.model_dump() for item in results],
    }
    return invoke_structured(
        llm,
        RestaurantWebEvidenceBatch,
        WEB_RESTAURANT_PROMPT,
        payload,
    ).restaurants


def _strict_dict(preferences: UserPreferences) -> dict:
    strict = preferences.strict_constraints
    if hasattr(strict, "model_dump"):
        return strict.model_dump(exclude_none=True)
    return dict(strict or {})


def _enrich_vietnamese_dish(candidate: dict) -> dict:
    record = dict(candidate.get("record") or {})
    if not record.get("cuisine") and record.get("source_dataset") in {FOODLENS_SOURCE, STREETFOOD_SOURCE}:
        record["cuisine"] = "Vietnamese"
    candidate = dict(candidate)
    candidate["record"] = record
    return candidate


def _explicit_cuisine_conflict(candidate: dict, preferences: UserPreferences) -> bool:
    wanted = {str(value).casefold() for value in preferences.cuisine if value}
    if not wanted:
        return False
    actual = candidate.get("record", {}).get("cuisine")
    if not actual:
        return False
    actual_values = actual if isinstance(actual, list) else [actual]
    return not any(str(value).casefold() in wanted for value in actual_values)


def retrieve_candidates(
    store,
    query: str,
    preferences: UserPreferences,
    *,
    candidate_depth: int = 20,
    retrieval_k: int = 3,
    web_search_client=None,
    web_candidate_extractor=None,
) -> list[dict]:
    intent = classify_intent(query)
    kind = {"recipe": "recipes", "dish": "dish", "restaurant": "restaurants"}[intent]
    all_preferences = preferences.model_dump(exclude_none=True)
    soft_prefs = {
        key: value
        for key, value in all_preferences.items()
        if key in {"cuisine", "budget", "price", "location", "atmosphere"} and value
    }
    strict = _strict_dict(preferences)

    dish_terms: list[str] = []
    if intent == "restaurant" and store.has_collection("dish_text"):
        for item in store.search("dish", query, {}, k=min(candidate_depth, 5), modality="text"):
            record = item.get("record") or {}
            term = record.get("name_vi") or record.get("name")
            if term:
                dish_terms.append(str(term))
            elif item.get("document"):
                dish_terms.append(str(item["document"]).split("|")[0].strip())

    search_preferences = {"strict_constraints": strict} if strict else {}
    text_results = store.search(
        kind,
        query,
        search_preferences,
        k=candidate_depth,
        modality="text",
        strict_only=bool(strict),
    )
    image_results = store.search(
        kind,
        query,
        search_preferences,
        k=candidate_depth,
        modality="image",
        strict_only=bool(strict),
    )
    if intent == "dish":
        text_results = [_enrich_vietnamese_dish(item) for item in text_results]
        image_results = [_enrich_vietnamese_dish(item) for item in image_results]

    fused = fuse_rank(text_results, image_results)
    ranked = rerank_candidates(fused, soft_prefs, dish_terms=dish_terms)

    if intent == "restaurant" and preferences.cuisine:
        ranked = [item for item in ranked if not _explicit_cuisine_conflict(item, preferences)]

    exact = [item for item in ranked if item.get("match_type") == "exact"]
    selected = (exact if exact else ranked)[: min(retrieval_k, 3)]

    if web_search_client is None or not should_use_external_search(query, selected, preferences):
        return selected

    try:
        web_results = web_search_client.search(query, max_results=5)
    except Exception:
        return selected

    evidence = [web_result_to_evidence(item).model_dump() for item in web_results]
    external_candidates: list[dict] = []
    if intent == "restaurant" and web_candidate_extractor is not None:
        try:
            extracted = web_candidate_extractor(web_results, query)
        except Exception:
            extracted = []
        for item in extracted:
            candidate = restaurant_evidence_to_candidate(item, soft_prefs)
            if candidate is not None and not _explicit_cuisine_conflict(candidate, preferences):
                external_candidates.append(candidate)

    merged = deduplicate_candidates(external_candidates + selected)
    merged.sort(
        key=lambda item: (
            item.get("match_type") != "exact",
            -float(item.get("ranking_score", 0.0)),
        )
    )
    merged = merged[: min(retrieval_k, 3)]
    if evidence:
        for candidate in merged:
            candidate.setdefault("external_evidence", evidence)
    return merged


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
    web_search_client = (
        TavilyWebSearchClient(settings.tavily_api_key)
        if settings.enable_web_search and settings.tavily_api_key
        else None
    )

    def retrieve(query: str, preferences: UserPreferences) -> list[dict]:
        return retrieve_candidates(
            store,
            query,
            preferences,
            candidate_depth=settings.candidate_depth,
            retrieval_k=settings.retrieval_k,
            web_search_client=web_search_client,
            web_candidate_extractor=(
                (lambda results, q: extract_restaurant_web_candidates(results, q, llm))
                if web_search_client is not None
                else None
            ),
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
            return "I couldn't find a relevant candidate in the indexed data or current web evidence that satisfies the mandatory constraints in your request."
        return "I couldn't find a sufficiently relevant restaurant, dish, or recipe in the indexed data or current web evidence for this request."
    candidates = {str(item["id"]): item for item in result.get("candidates", [])}
    response = result["recommendations"]
    lines = []
    for recommendation in sorted(response.recommendations, key=lambda item: item.rank)[:3]:
        candidate = candidates.get(recommendation.candidate_id)
        if candidate is None:
            raise ValueError(
                f"Recommendation {recommendation.candidate_id!r} is not present in workflow candidates"
            )
        record = candidate.get("record", {})
        name = record.get("name") or record.get("name_vi") or record.get("title") or recommendation.candidate_id
        note = ""
        if candidate.get("match_type") == "relaxed":
            missed = ", ".join(candidate.get("preference_mismatches", []))
            note = f" (closest match; differs on {missed})" if missed else " (closest match)"
        source_url = record.get("source_url") or candidate.get("source_url")
        source = f" [Source]({source_url})" if source_url else ""
        lines.append(f"{recommendation.rank}. **{name}** — {recommendation.reason}{note}{source}")
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
        gr.Markdown("Ask for a restaurant, dish, or recipe using cuisine, budget, atmosphere, location, or dietary preferences.")
        gr.ChatInterface(fn=chat)
    return app
