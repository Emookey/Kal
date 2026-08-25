"""Authentication and confinement contract for the synthetic Beepy service."""

import pytest

from src.beepy_service_auth import (
    beepy_service_route_allowed,
    resolve_api_token_identity,
)
from src.retrieval_policy import (
    BEEPY_SERVICE_SCOPES,
    BEEPY_SERVICE_TOKEN_PROFILE,
    KalConfigurationError,
    RetrievalContext,
    beepy_service_principal,
    shared_principal,
    validate_beepy_service_configuration,
)


def _reject_human_lookup(_owner):
    raise AssertionError("synthetic service identity must not require a human account")


def test_synthetic_identity_authenticates_without_human_account(monkeypatch):
    monkeypatch.delenv("KAL_BEEPY_SERVICE_PRINCIPAL", raising=False)
    monkeypatch.delenv("ODYSSEUS_BEEPY_SERVICE_PRINCIPAL", raising=False)
    identity = resolve_api_token_identity(
        "service:beepy",
        "chat,beepy:shared-rag",
        BEEPY_SERVICE_TOKEN_PROFILE,
        _reject_human_lookup,
    )
    assert identity is not None
    assert identity.owner == "service:beepy"
    assert identity.scopes == list(BEEPY_SERVICE_SCOPES)
    assert identity.profile == BEEPY_SERVICE_TOKEN_PROFILE
    assert identity.beepy_service is True


@pytest.mark.parametrize(
    ("owner", "scopes", "profile"),
    [
        ("alice", "chat,beepy:shared-rag", BEEPY_SERVICE_TOKEN_PROFILE),
        ("service:beepy", "chat", BEEPY_SERVICE_TOKEN_PROFILE),
        ("service:beepy", "chat,beepy:shared-rag,documents:read", BEEPY_SERVICE_TOKEN_PROFILE),
        ("service:beepy", "beepy:shared-rag,chat", BEEPY_SERVICE_TOKEN_PROFILE),
        ("service:beepy", "chat,beepy:shared-rag", None),
        ("service:beepy", "chat,beepy:shared-rag", "unknown"),
    ],
)
def test_service_owner_profile_scope_conflicts_fail_closed(owner, scopes, profile):
    assert resolve_api_token_identity(owner, scopes, profile, lambda value: str(value)) is None


def test_human_token_resolution_remains_compatible():
    identity = resolve_api_token_identity("Alice", "chat", None, lambda _value: "alice")
    assert identity is not None
    assert identity.owner == "alice"
    assert identity.scopes == ["chat"]
    assert identity.beepy_service is False


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/chat"),
        ("POST", "/api/chat_stream"),
        ("POST", "/api/session"),
        ("GET", "/api/sessions"),
        ("DELETE", "/api/session/example"),
        ("GET", "/api/memory"),
        ("GET", "/api/personal/documents"),
        ("GET", "/api/tools"),
        ("GET", "/api/agents"),
        ("GET", "/api/admin"),
        ("POST", "/api/webhooks"),
    ],
)
def test_service_token_is_denied_on_generic_routes(method, path):
    assert beepy_service_route_allowed(method, path) is False


def test_service_token_allows_only_exact_dedicated_methods():
    assert beepy_service_route_allowed("GET", "/api/integrations/beepy/capabilities")
    assert beepy_service_route_allowed("POST", "/api/integrations/beepy/technical-chat")
    assert not beepy_service_route_allowed("POST", "/api/integrations/beepy/capabilities")
    assert not beepy_service_route_allowed("GET", "/api/integrations/beepy/technical-chat")


def test_service_retrieval_is_shared_only_and_delete_is_denied():
    context = RetrievalContext.service(beepy_service_principal())
    assert context.allows_metadata({"owner": shared_principal()})
    assert not context.allows_metadata({"owner": "alice"})
    assert not context.allows_metadata({"owner": None})
    assert not context.allows_metadata({"owner": "service:other"})
    assert not context.allows_delete_metadata({"owner": shared_principal()})


@pytest.mark.parametrize(
    ("canonical_name", "legacy_name", "resolver", "value"),
    [
        ("KAL_BEEPY_SERVICE_PRINCIPAL", "ODYSSEUS_BEEPY_SERVICE_PRINCIPAL", beepy_service_principal, "service:beepy-ci"),
        ("KAL_SHARED_RAG_PRINCIPAL", "ODYSSEUS_SHARED_RAG_PRINCIPAL", shared_principal, "shared-ci"),
    ],
)
def test_canonical_and_legacy_configuration_alias_contract(
    monkeypatch, canonical_name, legacy_name, resolver, value
):
    monkeypatch.setenv(canonical_name, value)
    monkeypatch.delenv(legacy_name, raising=False)
    assert resolver() == value

    monkeypatch.delenv(canonical_name, raising=False)
    monkeypatch.setenv(legacy_name, value)
    assert resolver() == value

    monkeypatch.setenv(canonical_name, f" {value} ")
    assert resolver() == value

    monkeypatch.setenv(legacy_name, value + "-conflict")
    with pytest.raises(KalConfigurationError) as exc:
        resolver()
    assert canonical_name in str(exc.value)
    assert legacy_name in str(exc.value)
    assert value + "-conflict" not in str(exc.value)


def test_service_and_shared_owner_configuration_cannot_collapse(monkeypatch):
    monkeypatch.setenv("KAL_BEEPY_SERVICE_PRINCIPAL", "service:beepy")
    monkeypatch.setenv("KAL_SHARED_RAG_PRINCIPAL", "service:beepy")
    monkeypatch.delenv("ODYSSEUS_BEEPY_SERVICE_PRINCIPAL", raising=False)
    monkeypatch.delenv("ODYSSEUS_SHARED_RAG_PRINCIPAL", raising=False)
    with pytest.raises(KalConfigurationError) as exc:
        validate_beepy_service_configuration()
    assert "must be distinct" in str(exc.value)
