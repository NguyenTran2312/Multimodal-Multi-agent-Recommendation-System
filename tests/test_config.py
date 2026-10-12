from src.config import Settings


def test_tavily_key_optional_and_web_search_defaults_false(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("ENABLE_WEB_SEARCH", raising=False)
    settings = Settings.from_env()
    assert settings.tavily_api_key is None
    assert settings.enable_web_search is False


def test_web_search_flag_parses_truthy_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("TAVILY_API_KEY", " tav ")
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "YES")
    settings = Settings.from_env()
    assert settings.tavily_api_key == "tav"
    assert settings.enable_web_search is True
