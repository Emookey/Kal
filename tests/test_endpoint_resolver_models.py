"""Tests for endpoint_resolver — endpoint/model selection and enabled-model filtering."""
import json

from src.endpoint_resolver import (
    _first_chat_model,
    _endpoint_hidden_models,
    _endpoint_enabled_models,
)


class _Ep:
    """Minimal ModelEndpoint stand-in for the model-picking helpers."""
    def __init__(self, cached=None, hidden=None):
        self.cached_models = json.dumps(cached) if cached is not None else None
        self.hidden_models = json.dumps(hidden) if hidden is not None else None


class TestFirstChatModel:
    def test_skips_embedding_and_tts(self):
        models = ["text-embedding-ada-002", "whisper-large-v3", "gpt-4o"]
        assert _first_chat_model(models) == "gpt-4o"

    def test_falls_back_to_first_when_all_non_chat(self):
        assert _first_chat_model(["whisper-large-v3"]) == "whisper-large-v3"

    def test_empty(self):
        assert _first_chat_model([]) is None


class TestEnabledModels:
    def test_excludes_hidden(self):
        # The Groq repro: 16 models, only gpt-oss-120b enabled.
        cached = [
            "openai/gpt-oss-safeguard-20b", "canopylabs/orpheus-arabic-saudi",
            "whisper-large-v3", "openai/gpt-oss-120b",
        ]
        hidden = [
            "openai/gpt-oss-safeguard-20b", "canopylabs/orpheus-arabic-saudi",
            "whisper-large-v3",
        ]
        ep = _Ep(cached=cached, hidden=hidden)
        assert _endpoint_enabled_models(ep) == ["openai/gpt-oss-120b"]

    def test_no_hidden_returns_all(self):
        ep = _Ep(cached=["a", "b"], hidden=None)
        assert _endpoint_enabled_models(ep) == ["a", "b"]

    def test_picker_never_selects_disabled_model(self):
        # Regression: a disabled model listed first must not be auto-picked.
        cached = ["canopylabs/orpheus-arabic-saudi", "openai/gpt-oss-120b"]
        hidden = ["canopylabs/orpheus-arabic-saudi"]
        ep = _Ep(cached=cached, hidden=hidden)
        assert _first_chat_model(_endpoint_enabled_models(ep)) == "openai/gpt-oss-120b"

    def test_stale_configured_model_is_discarded(self):
        # A configured model that's been disabled is dropped, falling through
        # to the first enabled chat model.
        ep = _Ep(
            cached=["canopylabs/orpheus-arabic-saudi", "openai/gpt-oss-120b"],
            hidden=["canopylabs/orpheus-arabic-saudi"],
        )
        configured = "canopylabs/orpheus-arabic-saudi"
        if configured in _endpoint_hidden_models(ep):
            configured = ""
        if not configured:
            configured = _first_chat_model(_endpoint_enabled_models(ep))
        assert configured == "openai/gpt-oss-120b"


def test_service_endpoint_resolution_requires_shared_provider_owner(monkeypatch):
    from types import SimpleNamespace
    from src import endpoint_resolver as resolver
    from src import settings as settings_mod

    endpoint = SimpleNamespace(
        base_url="http://configured.invalid/v1",
        api_key=None,
        provider_auth_id=None,
        hidden_models=None,
        cached_models='["kal-owned-model"]',
        pinned_models=None,
    )

    class Query:
        def __init__(self):
            self.filter_calls = []

        def filter(self, *clauses):
            self.filter_calls.append(clauses)
            return self

        def first(self):
            return endpoint

    query = Query()
    db = SimpleNamespace(query=lambda _model: query, close=lambda: None)
    monkeypatch.setattr(resolver, "SessionLocal", lambda: db)
    monkeypatch.setattr(
        settings_mod,
        "load_settings",
        lambda: {"default_endpoint_id": "shared-endpoint", "default_model": "kal-owned-model"},
    )
    monkeypatch.setattr(
        settings_mod,
        "get_user_setting",
        lambda key, _owner, default="": {
            "default_endpoint_id": "shared-endpoint",
            "default_model": "kal-owned-model",
        }.get(key, default),
    )
    monkeypatch.setattr(resolver, "resolve_endpoint_runtime", lambda _ep, owner=None: (_ep.base_url, None))
    monkeypatch.setattr(resolver, "build_chat_url", lambda base: base + "/chat")
    monkeypatch.setattr(resolver, "build_headers", lambda _key, _base: {})

    resolved = resolver.resolve_endpoint("default", owner=None, shared_only=True)

    assert resolved[1] == "kal-owned-model"
    assert len(query.filter_calls) == 2
    assert "owner IS NULL" in str(query.filter_calls[1][0])
