from types import SimpleNamespace

from src.retrieval_policy import (
    BEEPY_SHARED_RAG_SCOPE,
    RetrievalContext,
    coerce_retrieval_context,
    is_beepy_service_request,
    shared_principal,
)


def test_user_private_shared_other_matrix():
    context = RetrievalContext.user("alice")
    assert context.allows_owner("alice")
    assert context.allows_owner(shared_principal())
    assert not context.allows_owner("bob")
    assert not context.allows_owner(None)


def test_delete_policy_is_action_specific_and_fail_closed():
    alice = RetrievalContext.user("alice")
    assert alice.allows_delete_metadata({"owner": "alice"})
    assert not alice.allows_delete_metadata({"owner": "bob"})
    assert not alice.allows_delete_metadata({"owner": shared_principal()})
    assert not alice.allows_delete_metadata({})

    service = RetrievalContext.service("service:beepy")
    assert not service.allows_delete_metadata({"owner": shared_principal()})
    assert not service.allows_delete_metadata({"owner": "alice"})
    assert not RetrievalContext.anonymous().allows_delete_metadata({})

    system = RetrievalContext.system("reconciliation-job")
    assert system.allows_delete_metadata({"owner": "alice"})
    assert system.allows_delete_metadata({"owner": shared_principal()})
    assert system.allows_delete_metadata({})


def test_owner_none_is_anonymous_not_unrestricted():
    context = coerce_retrieval_context(owner=None)
    assert context.allows_owner(None)
    assert not context.allows_owner("alice")
    assert not context.allows_owner(shared_principal())


def test_service_sees_only_explicit_shared_principal():
    context = RetrievalContext.service("service:beepy")
    assert context.allows_owner(shared_principal())
    assert not context.allows_owner("service:beepy")
    assert not context.allows_owner("alice")
    assert not context.allows_owner(None)


def test_system_context_is_explicitly_broad():
    context = RetrievalContext.system("reconciliation-job")
    assert context.allows_owner("alice")
    assert context.allows_owner("bob")
    assert context.allows_owner(shared_principal())
    assert context.allows_owner(None)


def test_empty_authenticated_identity_fails_closed():
    try:
        RetrievalContext.user("")
    except ValueError as exc:
        assert "explicit owner" in str(exc)
    else:
        raise AssertionError("empty authenticated owner was accepted")


def test_beepy_request_requires_explicit_scope_not_token_order_or_name():
    request = SimpleNamespace(state=SimpleNamespace(
        api_token=True,
        api_token_scopes=["chat", BEEPY_SHARED_RAG_SCOPE],
    ))
    assert is_beepy_service_request(request)
    request.state.api_token_scopes = ["chat"]
    assert not is_beepy_service_request(request)


def test_vector_semantic_keyword_and_scan_paths_apply_same_policy(monkeypatch, tmp_path):
    from tests.helpers.embedding_lanes import FakeChroma, FakeEmbedder, patch_chroma
    from src import embedding_lanes as lanes
    from src.rag_vector import VectorRAG

    fake = FakeChroma()
    patch_chroma(monkeypatch, fake)
    monkeypatch.setattr(lanes, "_build_custom_client", lambda: None)
    monkeypatch.setattr(
        lanes,
        "_build_fastembed_client",
        lambda: FakeEmbedder(384, "mini", "local://fake"),
    )
    rag = VectorRAG(persist_directory=str(tmp_path))
    for owner, label in (
        ("alice", "alice private"),
        ("bob", "bob private"),
        (shared_principal(), "approved shared"),
        (None, "legacy unowned"),
    ):
        metadata = {"source": label}
        if owner is not None:
            metadata["owner"] = owner
        assert rag.add_document(f"policy keyword {label}", metadata)

    user_docs = {row["document"] for row in rag.search("policy keyword", 10, owner="alice")}
    assert user_docs == {"policy keyword alice private", "policy keyword approved shared"}
    service_docs = {
        row["document"]
        for row in rag.scan_accessible(access_context=RetrievalContext.service("service:beepy"))
    }
    assert service_docs == {"policy keyword approved shared"}
    anonymous_docs = {row["document"] for row in rag.search("policy keyword", 10)}
    assert anonymous_docs == {"policy keyword legacy unowned"}

    for owner in ("alice", "bob", shared_principal(), None):
        metadata = {"source": "shared-source-id"}
        if owner is not None:
            metadata["owner"] = owner
        assert rag.add_document(f"delete scope {owner}", metadata)

    assert rag.delete_by_source("shared-source-id", owner="alice") == 1
    survivors = [
        row["metadata"].get("owner")
        for row in rag.scan_accessible(access_context=RetrievalContext.system("test"))
        if row["metadata"].get("source") == "shared-source-id"
    ]
    assert set(survivors) == {"bob", shared_principal(), None}

    assert rag.delete_by_source("shared-source-id", owner="bob") == 1
    assert rag.delete_by_source(
        "shared-source-id",
        access_context=RetrievalContext.service("service:beepy"),
    ) == 0
    assert rag.delete_by_source("shared-source-id") == 0
    assert rag.delete_by_source(
        "shared-source-id",
        access_context=RetrievalContext.system("test-cleanup"),
    ) == 2
    assert not [
        row
        for row in rag.scan_accessible(access_context=RetrievalContext.system("test"))
        if row["metadata"].get("source") == "shared-source-id"
    ]
