from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class UserPreferences(BaseModel):
    model_config = ConfigDict(strict=True)

    cuisine: list[str] = Field(default_factory=list)
    budget: str | None = None
    atmosphere: list[str] = Field(default_factory=list)
    dietary_preferences: list[str] = Field(default_factory=list)
    location: str | None = None


class ProfileOutput(BaseModel):
    summary: str
    priorities: list[str] = Field(default_factory=list)


class AnalysisOutput(BaseModel):
    summary: str
    signals: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    candidate_id: str
    reason: str
    rank: int = Field(ge=1)


class RecommendationResponse(BaseModel):
    recommendations: list[Recommendation] = Field(default_factory=list)
