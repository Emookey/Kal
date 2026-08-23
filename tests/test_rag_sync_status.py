from unittest.mock import MagicMock, patch

from routes.document_helpers import sync_document_to_rag
from src.rag_sync_status import clear_statuses_for_tests, get_status


def setup_function():
    clear_statuses_for_tests()


def test_sqlite_authority_reports_pending_when_vector_unavailable():
    with patch("src.rag_singleton.get_rag_manager", return_value=None):
        assert not sync_document_to_rag("doc-1", "authoritative text", "Title", "alice")
    status = get_status("document:doc-1")
    assert status.state == "pending"
    assert status.content_hash
    assert "retry" in status.detail


def test_atomic_sync_indexes_one_record_per_block_with_source_hash_metadata():
    rag = MagicMock(healthy=True)
    rag.delete_by_source.return_value = 0
    rag.add_document.return_value = True
    content = """DEVICE_HARDWARE_RECORD
Record_ID: REC-001
Device_ID: ACME-WS-001
RAM_GB: 8
END_DEVICE_HARDWARE_RECORD
DEVICE_RISK_RECORD
Record_ID: REC-002
Device_ID: ACME-WS-001
Performance_Risk: High
END_DEVICE_RISK_RECORD
"""
    with patch("src.rag_singleton.get_rag_manager", return_value=rag):
        assert sync_document_to_rag("doc-2", content, "client-rag-index", "alice")
    assert rag.add_document.call_count == 2
    first_metadata = rag.add_document.call_args_list[0].args[1]
    assert first_metadata["record_id"] == "REC-001"
    assert first_metadata["source"] == "document:doc-2"
    assert first_metadata["owner"] == "alice"
    assert len(first_metadata["content_hash"]) == 64
    assert get_status("document:doc-2").state == "indexed"


def test_malformed_or_partial_index_is_degraded_not_silently_successful():
    rag = MagicMock(healthy=True)
    rag.delete_by_source.return_value = 1
    with patch("src.rag_singleton.get_rag_manager", return_value=rag):
        assert not sync_document_to_rag(
            "doc-3", "DEVICE_RECORD\nDevice_ID: ACME-WS-001", "bad-rag-index", "alice"
        )
    assert rag.add_document.call_count == 0
    assert get_status("document:doc-3").state == "degraded"


def test_partial_vector_failure_records_degraded_with_indexed_count():
    rag = MagicMock(healthy=True)
    rag._split_into_chunks.return_value = ["one", "two"]
    rag.add_document.side_effect = [True, False]
    with patch("src.rag_singleton.get_rag_manager", return_value=rag):
        assert not sync_document_to_rag("doc-4", "one two", "ordinary", "alice")
    status = get_status("document:doc-4")
    assert status.state == "degraded"
    assert status.indexed_count == 1
