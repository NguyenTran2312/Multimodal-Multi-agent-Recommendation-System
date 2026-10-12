from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictConstraints(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    cuisine: list[str] | None = None
    budget: str | None = None
    price: str | None = None
    location: str | None = None
    dietary_preferences: list[str] | None = None


class UserPreferences(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    cuisine: list[str] = Field(default_factory=list)
    budget: str | None = None
    atmosphere: list[str] = Field(default_factory=list)
    dietary_preferences: list[str] = Field(default_factory=list)
    location: str | None = None
    strict_constraints: StrictConstraints = Field(default_factory=StrictConstraints)


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