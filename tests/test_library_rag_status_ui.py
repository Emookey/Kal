from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_library_rag_control_remains_visible_and_configurable():
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    index = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "'rag-toggle-btn':      '#overflow-rag-btn'" in app_js
    assert 'data-ui-key="rag-toggle-btn"' in index
    assert "Library RAG" in index


def test_library_surfaces_degraded_state_and_safe_retry_action():
    source = (ROOT / "static" / "js" / "documentLibrary.js").read_text(encoding="utf-8")
    assert "doclib-rag-state" in source
    assert "Retry RAG" in source
    assert "/reconcile-rag" in source
    assert "textContent = 'Retry RAG'" in source

