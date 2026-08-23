"""Static lifecycle matrix guarding every authoritative Library mutation hook."""

import inspect

from routes import document_routes


def test_complete_library_rag_lifecycle_hooks_remain_wired():
    source = inspect.getsource(document_routes.setup_document_routes)
    required = {
        "create": '_sync_rag_for_doc(doc, "new")',
        "import": '_sync_rag_for_doc(doc, "imported PDF")',
        "archive": '_delete_rag_for_doc(doc_id, "archived")',
        "unarchive": '_sync_rag_for_doc(doc, "unarchived")',
        "ocr": '_sync_rag_for_doc(doc, "re-extracted PDF")',
        "update": '_sync_rag_for_doc(doc, "updated")',
        "patch": '_sync_rag_for_doc(doc, "patched")',
        "restore": '_sync_rag_for_doc(doc, "restored version")',
        "tidy": '_delete_rag_for_doc(doc.id, "tidied")',
        "ai_tidy": '_delete_rag_for_doc(doc.id, "AI tidied")',
        "delete": '_delete_rag_for_doc(doc_id, "deleted")',
        "retry": '"/api/document/{doc_id}/reconcile-rag"',
    }
    missing = {name: hook for name, hook in required.items() if hook not in source}
    assert not missing
    assert '"rag_index_degraded": rag_degraded' in source


def test_library_commits_sqlite_before_best_effort_rag_sync():
    source = inspect.getsource(document_routes.setup_document_routes)
    update_start = source.index("async def update_document")
    patch_start = source.index("async def patch_document", update_start)
    update_block = source[update_start:patch_start]
    assert update_block.index("db.commit()") < update_block.index('_sync_rag_for_doc(doc, "updated")')
