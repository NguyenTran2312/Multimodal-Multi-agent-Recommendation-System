from __future__ import annotations

from typing import Callable, TypedDict

from src.agents.schemas import RecommendationResponse, UserPreferences


class WorkflowError(RuntimeError):
    """Controlled application-level workflow failure."""


class RecommendationState(TypedDict, total=False):
    query: str
    preferences: UserPreferences
    profile: object
    candidates: list[dict]
    trends: object
    style: object
    nutrition: object
    recommendations: RecommendationResponse
    status: str


class RecommendationWorkflow:
    def __init__(
        self,
        *,
        preference_extractor: Callable[[str], UserPreferences],
        retrieve: Callable[[str, UserPreferences], list[dict]],
        profile_agent: Callable[[UserPreferences], object],
        trend_agent: Callable[[list[dict], UserPreferences], object],
        style_agent: Callable[[list[dict], UserPreferences], object],
        nutrition_agent: Callable[[list[dict], UserPreferences], object],
        recommender: Callable[[list[dict], dict, UserPreferences], RecommendationResponse],
    ):
        self.preference_extractor = preference_extractor
        self.retrieve = retrieve
        self.profile_agent = profile_agent
        self.trend_agent = trend_agent
        self.style_agent = style_agent
        self.nutrition_agent = nutrition_agent
        self.recommender = recommender

    def recommend(self, query: str) -> dict:
        try:
            preferences = self.preference_extractor(query)
            candidates = self.retrieve(query, preferences)
            if not candidates:
                return {"status": "no_match", "query": query, "candidates": []}

            profile = self.profile_agent(preferences)
            analyses = {
                "trends": self.trend_agent(candidates, preferences),
                "style": self.style_agent(candidates, preferences),
                "nutrition": self.nutrition_agent(candidates, preferences),
            }
            recommendations = self.recommender(candidates, analyses, preferences)
            allowed = {str(candidate["id"]) for candidate in candidates}
            for recommendation in recommendations.recommendations:
                if recommendation.candidate_id not in allowed:
                    raise WorkflowError(
                        f"Recommendation candidate {recommendation.candidate_id!r} was not retrieved"
                    )
            return {
                "status": "ok",
                "query": query,
                "preferences": preferences,
                "profile": profile,
                "candidates": candidates,
                "analyses": analyses,
                "recommendations": recommendations,
            }
        except WorkflowError:
            raise
        except Exception as exc:
            raise WorkflowError(f"Recommendation workflow failed: {exc}") from exc


class CompiledGraphWorkflow:
    def __init__(self, graph):
        self.graph = graph

    def recommend(self, query: str) -> dict:
        try:
            return self.graph.invoke({"query": query})
        except WorkflowError:
            raise
        except Exception as exc:
            raise WorkflowError(f"Recommendation workflow failed: {exc}") from exc


def build_recommendation_graph(workflow: RecommendationWorkflow):
    """Build a LangGraph representation of the workflow for runtime orchestration."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise ImportError(
            "LangGraph is required at runtime. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    graph = StateGraph(RecommendationState)

    def extract(state: RecommendationState):
        return {"preferences": workflow.preference_extractor(state["query"])}

    def retrieve(state: RecommendationState):
        return {
            "candidates": workflow.retrieve(state["query"], state["preferences"]),
        }

    def profile(state: RecommendationState):
        if not state["candidates"]:
            return {"status": "no_match"}
        return {"profile": workflow.profile_agent(state["preferences"])}

    def trends(state: RecommendationState):
        if not state["candidates"]:
            return {}
        return {"trends": workflow.trend_agent(state["candidates"], state["preferences"])}

    def style(state: RecommendationState):
        if not state["candidates"]:
            return {}
        return {"style": workflow.style_agent(state["candidates"], state["preferences"])}

    def nutrition(state: RecommendationState):
        if not state["candidates"]:
            return {}
        return {"nutrition": workflow.nutrition_agent(state["candidates"], state["preferences"])}

    def final(state: RecommendationState):
        if not state["candidates"]:
            return {"status": "no_match"}
        analyses = {
            "trends": state.get("trends"),
            "style": state.get("style"),
            "nutrition": state.get("nutrition"),
        }
        return {
            "status": "ok",
            "recommendations": workflow.recommender(
                state["candidates"], analyses, state["preferences"]
            ),
        }

    graph.add_node("extract", extract)
    graph.add_node("retrieve", retrieve)
    graph.add_node("profile", profile)
    graph.add_node("trends", trends)
    graph.add_node("style", style)
    graph.add_node("nutrition", nutrition)
    graph.add_node("final", final)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "retrieve")
    analysis_nodes = ["profile", "trends", "style", "nutrition"]
    for node in analysis_nodes:
        graph.add_edge("retrieve", node)
    graph.add_edge(analysis_nodes, "final")
    graph.add_edge("final", END)
    return graph.compile()
