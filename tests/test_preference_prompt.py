from src.ui.chatbot import PREFERENCE_PROMPT


def test_preference_prompt_preserves_soft_vs_strict_constraints_and_bilingual_food_terms():
    text = PREFERENCE_PROMPT.lower()
    assert "english and vietnamese" in text
    assert "soft preference" in text
    assert "strict_constraints" in text
    for term in ("must", "only", "chỉ", "bắt buộc"):
        assert term in text
    for unstated in ("location", "budget", "allergy", "dietary status"):
        assert unstated in text
    assert "empty lists or null" in text
