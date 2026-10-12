from src.agents.prompt_policy import (
    CULINARY_ROLE_POLICY,
    GROUNDING_POLICY,
    LANGUAGE_POLICY,
    USER_FACING_STYLE_POLICY,
    compose_prompt,
)


def test_shared_prompt_policy_defines_production_culinary_role():
    text = " ".join(
        [
            CULINARY_ROLE_POLICY,
            GROUNDING_POLICY,
            LANGUAGE_POLICY,
            USER_FACING_STYLE_POLICY,
        ]
    ).lower()
    assert "culinary" in text
    assert "restaurant" in text
    assert "recipe" in text
    assert "do not invent" in text or "never invent" in text
    assert "same language" in text
    assert "professional" in text
    assert "approachable" in text


def test_shared_prompt_policy_hides_internal_system_details():
    text = USER_FACING_STYLE_POLICY.lower()
    for term in (
        "ranking score",
        "fusion score",
        "modality score",
        "semantic score",
        "embedding",
        "candidate id",
        "vector search",
    ):
        assert term in text
    assert "do not mention" in text or "never mention" in text


def test_compose_prompt_includes_shared_rules_and_task_instruction():
    prompt = compose_prompt("TASK: Recommend only the supplied candidates.")
    assert CULINARY_ROLE_POLICY in prompt
    assert GROUNDING_POLICY in prompt
    assert LANGUAGE_POLICY in prompt
    assert USER_FACING_STYLE_POLICY in prompt
    assert "Recommend only the supplied candidates" in prompt
