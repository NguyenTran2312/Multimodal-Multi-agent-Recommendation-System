from src.agents.nutrition import PROMPT as NUTRITION_PROMPT
from src.agents.profile import PROMPT as PROFILE_PROMPT
from src.agents.recommender import PROMPT as RECOMMENDER_PROMPT
from src.agents.style import PROMPT as STYLE_PROMPT
from src.agents.trends import PROMPT as TRENDS_PROMPT


def test_profile_prompt_separates_hard_and_soft_preferences_without_persona_assumptions():
    text = PROFILE_PROMPT.lower()
    assert "hard constraint" in text
    assert "soft preference" in text
    assert "do not infer" in text or "do not invent" in text
    for term in ("age", "lifestyle", "health goal", "spending habit"):
        assert term in text


def test_trends_prompt_uses_observed_evidence_not_unsupported_popularity_claims():
    text = TRENDS_PROMPT.lower()
    assert "evidence" in text
    assert "rating" in text
    assert "review" in text
    for term in ("trending", "viral", "popular"):
        assert term in text
    assert "do not" in text or "never" in text


def test_style_prompt_is_grounded_in_available_descriptions_and_metadata():
    text = STYLE_PROMPT.lower()
    assert "atmosphere" in text
    assert "description" in text
    assert "image caption" in text
    for term in ("clip", "embedding", "modality score"):
        assert term in text


def test_nutrition_prompt_preserves_constraints_and_forbids_invented_numbers_or_medical_claims():
    text = NUTRITION_PROMPT.lower()
    assert "dietary constraint" in text
    for term in (
        "calorie",
        "macro",
        "allergen guarantee",
        "medical benefit",
        "health outcome",
    ):
        assert term in text
    assert "do not invent" in text or "never invent" in text


def test_recommender_prompt_requires_natural_grounded_user_facing_reasons():
    text = RECOMMENDER_PROMPT.lower()
    assert "provided candidate" in text
    assert "professional" in text
    assert "approachable" in text
    assert "same language" in text
    for term in (
        "ranking score",
        "fusion score",
        "modality score",
        "semantic score",
        "embedding",
        "candidate id",
    ):
        assert term in text
    assert "do not mention" in text or "never mention" in text
    assert "weak alternative" in text
