from src.structured_rag import retrieve_structured


def _record(record_type, record_id, text, **metadata):
    return {
        "id": record_id,
        "document": f"{record_type}\nRecord_ID: {record_id}\n{text}\nEND_{record_type}",
        "metadata": {
            "source": "document:test-index",
            "filename": "test-rag-index.md",
            "chunk_id": record_id,
            "atomic": "true",
            "record_type": record_type,
            "record_id": record_id,
            **metadata,
        },
    }


def test_exact_identifier_not_found_suppresses_semantic_substitution():
    records = [_record(
        "DEVICE_HARDWARE_RECORD", "REC-001", "Device_ID: ACME-WS-001",
        device_id="ACME-WS-001",
    )]
    semantic = [{**records[0], "similarity": 0.99}]
    outcome = retrieve_structured("Show ACME-WS-009", records, semantic)
    assert outcome.missing_identifiers == ["ACME-WS-009"]
    assert [row["metadata"]["record_type"] for row in outcome.results] == [
        "EXACT_ID_NOT_FOUND_RECORD"
    ]
    assert "ACME-WS-001" not in outcome.results[0]["document"]


def test_literal_fallback_is_exact_and_multi_id_selection_is_fair():
    records = [
        _record("DEVICE_HARDWARE_RECORD", "A-HW", "Device_ID: ACME-WS-001", device_id="ACME-WS-001"),
        _record("DEVICE_RISK_RECORD", "A-RISK", "Device_ID: ACME-WS-001", device_id="ACME-WS-001"),
        _record("DEVICE_HARDWARE_RECORD", "B-HW", "literal ACME-WS-002", device_id="different"),
        _record("DEVICE_RISK_RECORD", "B-RISK", "Device_ID: ACME-WS-002", device_id="ACME-WS-002"),
    ]
    outcome = retrieve_structured(
        "Compare ACME-WS-001 and ACME-WS-002 hardware risk", records, []
    )
    ids = [row["metadata"]["record_id"] for row in outcome.results]
    assert ids[:2] == ["A-HW", "B-HW"]
    assert ids[2:4] == ["A-RISK", "B-RISK"]


def test_person_application_resolution_and_user_only_identifier_carry():
    records = [
        _record(
            "USER_ASSIGNMENT_RECORD", "ASSIGN-1", "Assigned_User: Jamie Doe\nDevice_ID: ACME-WS-001",
            assigned_user="Jamie Doe", device_id="ACME-WS-001", client_id="CLIENT-001",
        ),
        _record(
            "DEVICE_APPLICATION_RECORD", "APP-1", "Application: Backup Pro\nDevice_ID: ACME-WS-001",
            application_name="Backup Pro", device_id="ACME-WS-001",
        ),
    ]
    history = [
        {"role": "user", "content": "Review ACME-WS-001"},
        {"role": "assistant", "content": "A similar system is EVIL-WS-999"},
    ]
    outcome = retrieve_structured(
        "Does Jamie Doe use Backup Pro, and compare it?", records, [], history=history
    )
    assert "ACME-WS-001" in outcome.identifiers
    assert "EVIL-WS-999" not in outcome.identifiers
    types = [row["metadata"]["record_type"] for row in outcome.results]
    assert "USER_ASSIGNMENT_RECORD" in types
    assert "DEVICE_APPLICATION_RECORD" in types


def test_deterministic_aggregate_is_complete_sorted_and_deduplicated():
    records = [
        _record("DEVICE_HARDWARE_RECORD", "R2", "RAM_GB: 8", device_id="ACME-WS-002", client_id="CLIENT-B", ram_gb="8"),
        _record("DEVICE_HARDWARE_RECORD", "R1", "RAM_GB: 4", device_id="ACME-WS-001", client_id="CLIENT-A", ram_gb="4"),
        _record("DEVICE_RISK_RECORD", "R1B", "RAM_GB: 4", device_id="ACME-WS-001", client_id="CLIENT-A", ram_gb="4"),
        _record("DEVICE_HARDWARE_RECORD", "R3", "RAM_GB: 32", device_id="ACME-WS-003", client_id="CLIENT-A", ram_gb="32"),
    ]
    outcome = retrieve_structured("List all devices with RAM under 16 GB", records, [])
    assert outcome.aggregate is not None
    text = outcome.results[0]["document"]
    assert "Match_Count: 2" in text
    assert text.count("Device_ID=ACME-WS-001") == 1
    assert text.index("ACME-WS-001") < text.index("ACME-WS-002")
    assert "ACME-WS-003" not in text


def test_complete_sop_assembly_and_authoritative_order_before_semantic_support():
    records = [
        _record("SOP_OUTPUT_RECORD", "SOP-OUT", "SOP_ID: SOP-MFA-001\nOutput: document result", sop_id="SOP-MFA-001"),
        _record("SOP_ACTION_RECORD", "SOP-ACT", "SOP_ID: SOP-MFA-001\nAction: reset MFA", sop_id="SOP-MFA-001"),
        _record("SOP_GUARDRAIL_RECORD", "SOP-GUARD", "SOP_ID: SOP-MFA-001\nGuardrail: verify identity", sop_id="SOP-MFA-001"),
        _record("SOP_TRIGGER_RECORD", "SOP-TRIG", "SOP_ID: SOP-MFA-001\nTrigger: MFA failure", sop_id="SOP-MFA-001"),
        _record("SOP_DECISION_RECORD", "SOP-DEC", "SOP_ID: SOP-MFA-001\nDecision: escalate", sop_id="SOP-MFA-001"),
    ]
    semantic = [{
        "id": "semantic",
        "document": "general MFA troubleshooting",
        "metadata": {"source": "document:general", "chunk_id": 0},
        "similarity": 0.9,
    }]
    outcome = retrieve_structured("What is the SOP procedure for MFA failure?", records, semantic)
    types = [row["metadata"].get("record_type") for row in outcome.results]
    assert types[:5] == [
        "SOP_TRIGGER_RECORD", "SOP_ACTION_RECORD", "SOP_GUARDRAIL_RECORD",
        "SOP_DECISION_RECORD", "SOP_OUTPUT_RECORD",
    ]
    assert outcome.results[-1]["metadata"]["grounding"] == "semantic_support"

