"""Focused current-identity and legacy-compatibility contract for Phase 3B."""

from pathlib import Path

from src import constants
from src.retrieval_policy import RetrievalContext, shared_principal


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_application_and_platform_names_are_distinct():
    assert constants.APP_NAME == "Kal"
    assert constants.PLATFORM_NAME == "Kalvin"
    assert constants.LEGACY_APP_NAME == "Odysseus"


def test_kal_data_dir_precedes_legacy_override(monkeypatch, tmp_path):
    canonical = tmp_path / "kal"
    legacy = tmp_path / "odysseus"
    monkeypatch.setenv("KAL_DATA_DIR", str(canonical))
    monkeypatch.setenv("ODYSSEUS_DATA_DIR", str(legacy))
    assert constants.resolve_data_dir() == str(canonical)


def test_legacy_data_dir_remains_supported(monkeypatch, tmp_path):
    legacy = tmp_path / "odysseus"
    monkeypatch.delenv("KAL_DATA_DIR", raising=False)
    monkeypatch.setenv("ODYSSEUS_DATA_DIR", str(legacy))
    assert constants.resolve_data_dir() == str(legacy)


def test_empty_overrides_retain_existing_default(monkeypatch, tmp_path):
    default = tmp_path / "default"
    monkeypatch.setenv("KAL_DATA_DIR", "")
    monkeypatch.setenv("ODYSSEUS_DATA_DIR", "")
    monkeypatch.setattr(constants, "get_default_data_dir", lambda: str(default))
    assert constants.resolve_data_dir() == str(default)


def test_primary_user_facing_surfaces_identify_as_kal():
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    login = (ROOT / "static" / "login.html").read_text(encoding="utf-8")
    manifest = (ROOT / "static" / "manifest.json").read_text(encoding="utf-8")
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

    assert "<title>Kal Chat</title>" in index
    assert '<span class="sidebar-brand-title">Kal</span>' in index
    assert 'placeholder="Message Kal..."' in index
    assert "<title>Kal — Login</title>" in login
    assert '"name": "Kal"' in manifest
    assert "'Kal Chat'" in app_js
    assert "Message Kal..." in app_js


def test_safe_external_display_metadata_uses_kal():
    oauth = (ROOT / "src" / "mcp_oauth.py").read_text(encoding="utf-8")
    endpoints = (ROOT / "src" / "endpoint_resolver.py").read_text(encoding="utf-8")
    assert 'client_name="Kal"' in oauth
    assert 'headers.setdefault("X-OpenRouter-Title", "Kal")' in endpoints


def test_legacy_runtime_and_persistent_identifiers_are_preserved():
    source_constants = (ROOT / "src" / "constants.py").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    auth_routes = (ROOT / "routes" / "auth_routes.py").read_text(encoding="utf-8")
    rag_vector = (ROOT / "src" / "rag_vector.py").read_text(encoding="utf-8")

    assert 'os.getenv("ODYSSEUS_DATA_DIR")' in source_constants
    assert "X-Odysseus-Internal-Token" in app
    assert 'SESSION_COOKIE = "odysseus_session"' in auth_routes
    assert 'COLLECTION_NAME = "odysseus_rag"' in rag_vector
    assert shared_principal() == "__goodwill_shared__"


def test_beepy_security_contract_is_unchanged():
    service = RetrievalContext.service("service:beepy")
    assert service.allows_owner(shared_principal())
    assert not service.allows_owner("alice")
    assert not service.allows_delete_metadata({"owner": shared_principal()})
    assert not service.allows_delete_metadata({"owner": "alice"})
