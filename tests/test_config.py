from src.config import Settings


def test_external_keys_optional_and_flags_default_false(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    monkeypatch.delenv("ENABLE_WEB_SEARCH", raising=False)
    monkeypatch.delenv("ENABLE_PLACES_SEARCH", raising=False)
    settings = Settings.from_env()
    assert settings.tavily_api_key is None
    assert settings.google_places_api_key is None
    assert settings.enable_web_search is False
    assert settings.enable_places_search is False


def test_external_flags_parse_truthy_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("TAVILY_API_KEY", " tav ")
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", " gp ")
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "YES")
    monkeypatch.setenv("ENABLE_PLACES_SEARCH", "on")
    settings = Settings.from_env()
    assert settings.tavily_api_key == "tav"
    assert settings.google_places_api_key == "gp"
    assert settings.enable_web_search is True
    assert settings.enable_places_search is True
