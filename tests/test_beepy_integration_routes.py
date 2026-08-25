"""Focused stateless Beepy capability and grounding response tests."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from routes import beepy_integration_routes as routes_mod
from src import beepy_service as service_mod
from src.retrieval_policy import (
    BEEPY_SERVICE_TOKEN_PROFILE,
    BEEPY_SHARED_RAG_SCOPE,
    shared_principal,
)


class FakeRag:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.contexts = []

    def search(self, _message, *, k, owner, access_context):
        if self.fail:
            raise RuntimeError("private datastore detail")
        self.contexts.append((k, owner, access_context))
        return []

    def scan_accessible(self, *, access_context):
        self.contexts.append(("scan", access_context))
        return []


def _result(owner, label, *, grounding="semantic_support"):
    return {
        "document": f"approved evidence from {label}",
        "similarity": 0.9,
        "metadata": {
            "owner": owner,
            "filename": label,
            "grounding": grounding,
        },
    }


def _handlers(monkeypatch, rag, results, capture):
    monkeypatch.setattr(
        service_mod,
        "retrieve_structured",
        lambda *_args, **_kwargs: SimpleNamespace(results=results),
    )
    monkeypatch.setattr(
        routes_mod,
        "resolve_endpoint",
        lambda prefix, owner=None, shared_only=False: (
            capture.update(selection=(prefix, owner, shared_only))
            or ("http://configured.invalid/v1/chat", "kal-owned-model", {"X-Test": "1"})
        ),
    )

    async def fake_llm(url, model, messages, **kwargs):
        capture.update(url=url, model=model, messages=messages, kwargs=kwargs)
        return "bounded answer"

    monkeypatch.setattr(routes_mod, "llm_call_async", fake_llm)
    router = routes_mod.setup_beepy_integration_routes(rag)
    handlers = {route.path: route.endpoint for route in router.routes}
    return handlers["/api/integrations/beepy/capabilities"], handlers["/api/integrations/beepy/technical-chat"]


def _request(*, exact_service=True):
    return SimpleNamespace(
        state=SimpleNamespace(
            api_token=True,
            api_token_owner="service:beepy" if exact_service else "alice",
            api_token_scopes=["chat", BEEPY_SHARED_RAG_SCOPE],
            api_token_profile=BEEPY_SERVICE_TOKEN_PROFILE,
        )
    )


def _call_chat(handler, message, **audit):
    return asyncio.run(
        handler(
            _request(),
            routes_mod.BeepyTechnicalRequest(message=message),
            beepy_actor=audit.get("beepy_actor"),
            correlation_id=audit.get("correlation_id"),
            business_operation=audit.get("business_operation"),
        )
    )


def test_grounded_response_is_stateless_shared_only_and_auditable(monkeypatch):
    rag = FakeRag()
    capture = {}
    results = [
        _result(shared_principal(), "/srv/internal/technical-guide.pdf", grounding="structured_exact"),
        _result("alice", "/private/alice-notes.txt"),
        _result(None, "/legacy/null-owner.txt"),
        _result("service:other", "/service/other.txt"),
    ]
    _, chat = _handlers(monkeypatch, rag, results, capture)
    body = _call_chat(
        chat,
        "How do I troubleshoot the shared platform?",
        beepy_actor="beepy-user-42",
        correlation_id="req-123",
        business_operation="technical-support",
    )

    assert body == {
        "contract_version": "beepy-kal.v1",
        "response": "bounded answer",
        "shared_knowledge_used": True,
        "grounding_status": "grounded",
        "sources": [{"label": "technical-guide.pdf", "grounding": "structured_exact"}],
        "request_id": "req-123",
    }
    assert capture["model"] == "kal-owned-model"
    assert capture["selection"] == ("default", None, True)
    assert capture["kwargs"]["temperature"] == 0.2
    assert capture["kwargs"]["max_tokens"] == 2048
    assert all("alice-notes" not in str(message) for message in capture["messages"])
    assert rag.contexts[0][1] == "service:beepy"
    assert rag.contexts[0][2].principal == "service:beepy"
    assert not hasattr(rag.contexts[0][2], "actor")


def test_no_match_and_degraded_grounding_states_are_explicit(monkeypatch):
    for rag, expected in ((FakeRag(), "no_match"), (FakeRag(fail=True), "degraded")):
        capture = {}
        _, chat = _handlers(monkeypatch, rag, [], capture)
        body = _call_chat(chat, "unmatched technical question")
        assert body["grounding_status"] == expected
        assert body["shared_knowledge_used"] is False
        assert body["sources"] == []
        assert body["request_id"]
        assert "private datastore detail" not in str(body)


def test_source_metadata_is_bounded_deduplicated_and_path_free(monkeypatch):
    results = [
        _result(shared_principal(), f"/private/root/source-{index}-" + "x" * 180)
        for index in range(10)
    ]
    capture = {}
    _, chat = _handlers(monkeypatch, FakeRag(), results, capture)
    body = _call_chat(chat, "bounded sources")
    sources = body["sources"]
    assert len(sources) == 5
    assert all(set(item) == {"label", "grounding"} for item in sources)
    assert all(len(item["label"]) <= 120 for item in sources)
    assert all("/" not in item["label"] and "\\" not in item["label"] for item in sources)
    assert "owner" not in str(body)
    assert "/private/root" not in str(body)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "session",
        "owner",
        "scopes",
        "attachments",
        "memory",
        "documents",
        "tools",
        "agent_mode",
        "workspace",
        "url_preprocessing",
        "use_web",
        "use_research",
        "image_generation",
        "webhook",
        "provider",
        "model",
        "endpoint_url",
        "api_key",
        "delete",
    ],
)
def test_privileged_and_stateful_request_fields_are_unreachable(monkeypatch, forbidden_field):
    with pytest.raises(ValidationError):
        routes_mod.BeepyTechnicalRequest.model_validate(
            {"message": "technical question", forbidden_field: True}
        )


def test_urls_are_plain_question_text_not_preprocessed(monkeypatch):
    capture = {}
    _, chat = _handlers(monkeypatch, FakeRag(), [], capture)
    message = "Explain this technical URL without fetching it: https://example.invalid/doc"
    _call_chat(chat, message)
    assert capture["messages"][-1] == {"role": "user", "content": message}


def test_capability_requires_exact_service_identity(monkeypatch):
    capabilities, _ = _handlers(monkeypatch, FakeRag(), [], {})
    with pytest.raises(HTTPException) as exc:
        capabilities(_request(exact_service=False))
    assert exc.value.status_code == 403
    assert capabilities(_request())["stateless"] is True


@pytest.mark.parametrize(
    ("header", "value"),
    [
        ("X-Kal-Beepy-Actor", "bad actor with spaces"),
        ("X-Kal-Request-ID", "x" * 129),
        ("X-Kal-Business-Operation", "bad/value"),
    ],
)
def test_audit_metadata_constraints_fail_closed(monkeypatch, header, value):
    _, chat = _handlers(monkeypatch, FakeRag(), [], {})
    keyword = {
        "X-Kal-Beepy-Actor": "beepy_actor",
        "X-Kal-Request-ID": "correlation_id",
        "X-Kal-Business-Operation": "business_operation",
    }[header]
    with pytest.raises(HTTPException) as exc:
        _call_chat(chat, "technical question", **{keyword: value})
    assert exc.value.status_code == 400
    assert value not in str(exc.value.detail)
