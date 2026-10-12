from __future__ import annotations

from src.agents._common import invoke_structured
from src.agents.prompt_policy import compose_prompt
from src.agents.schemas import AnalysisOutput, UserPreferences


PROMPT = compose_prompt(
    """
TASK: Analyze cuisine style, dining atmosphere, and visual fit for the retrieved candidates.

Ground atmosphere and visual observations only in supplied restaurant or recipe descriptions,
image captions, and structured metadata. Use user-facing terms such as quiet, lively, casual,
romantic, minimalist, family-style, upscale, colorful, or rustic only when the records support
them. Do not infer visual details that are absent from the input. Do not mention CLIP,
embedding or embeddings, modality score, or other image-retrieval implementation details.
"""
)


def analyze_style(candidates: list[dict], preferences: UserPreferences, llm) -> AnalysisOutput:
    return invoke_structured(
        llm, AnalysisOutput, PROMPT,
        {"candidates": candidates, "preferences": preferences.model_dump()},
    )
