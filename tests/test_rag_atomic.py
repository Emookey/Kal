import pytest

from src.rag_atomic import (
    MAX_INPUT_CHARS,
    AtomicRAGParseError,
    parse_atomic_rag_records,
)


def test_atomic_parser_recognizes_records_rules_and_metadata_deterministically():
    content = """preface ignored
DEVICE_HARDWARE_RECORD
Record_ID: REC-DEV-001
Client_ID: CLIENT-ACME-001
Device_ID: ACME-WS-001
RAM_GB: 8
Storage_Type: HDD
END_DEVICE_HARDWARE_RECORD

PRIORITY_RULE
Record_ID: RULE-PRI-001
SOP_ID: SOP-PATCH-001
Applies_To: CLIENT-ACME-001
Trigger: failed patch
END_PRIORITY_RULE
"""
    first = parse_atomic_rag_records(content)
    second = parse_atomic_rag_records(content)
    assert first == second
    assert [record.metadata["record_type"] for record in first] == [
        "DEVICE_HARDWARE_RECORD", "PRIORITY_RULE"
    ]
    assert first[0].metadata["device_id"] == "ACME-WS-001"
    assert first[0].metadata["ram_gb"] == "8"
    assert first[1].metadata["sop_id"] == "SOP-PATCH-001"
    assert len(first[0].metadata["content_hash"]) == 64


def test_atomic_parser_requires_matching_end_marker():
    with pytest.raises(AtomicRAGParseError, match="missing END_DEVICE_RECORD"):
        parse_atomic_rag_records("DEVICE_RECORD\nDevice_ID: ACME-WS-001\n")


def test_atomic_parser_rejects_nested_and_duplicate_records():
    with pytest.raises(AtomicRAGParseError, match="nested marker"):
        parse_atomic_rag_records(
            "DEVICE_RECORD\nOTHER_RECORD\nEND_OTHER_RECORD\nEND_DEVICE_RECORD"
        )
    duplicate = """DEVICE_RECORD
Record_ID: REC-DUP-001
END_DEVICE_RECORD
DEVICE_RECORD
Record_ID: REC-DUP-001
END_DEVICE_RECORD
"""
    with pytest.raises(AtomicRAGParseError, match="Duplicate"):
        parse_atomic_rag_records(duplicate)


def test_atomic_parser_bounds_input_and_does_not_restore_neighbor_chunks():
    with pytest.raises(AtomicRAGParseError, match="exceeds"):
        parse_atomic_rag_records("x" * (MAX_INPUT_CHARS + 1))
    record = parse_atomic_rag_records(
        "DEVICE_RECORD\nDevice_ID: ACME-WS-001\nEND_DEVICE_RECORD"
    )[0]
    assert "neighbor" not in record.metadata
    assert "chunk_number" not in record.metadata

