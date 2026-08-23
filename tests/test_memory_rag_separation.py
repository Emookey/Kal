from types import SimpleNamespace

from src.memory_policy import (
    MemoryOrigin,
    automatic_extraction_messages,
    is_business_or_rag_fact,
    should_auto_pin,
)


def test_automatic_extraction_input_is_user_only():
    messages = [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Alice owns ACME-WS-001"},
        SimpleNamespace(role="system", content="hidden"),
    ]
    kept = automatic_extraction_messages(messages)
    assert kept == [messages[0]]


def test_business_identifiers_inventory_fields_and_non_user_origins_are_rejected():
    assert is_business_or_rag_fact("Device_ID: ACME-WS-001")
    assert is_business_or_rag_fact("RAM_GB: 8")
    assert is_business_or_rag_fact("DEVICE_HARDWARE_RECORD")
    assert is_business_or_rag_fact("ordinary sentence", origin=MemoryOrigin.BUSINESS_DATA)
    assert not is_business_or_rag_fact("The user prefers concise answers")


def test_only_fallback_user_identity_is_auto_pinned():
    assert should_auto_pin(category="identity", fallback=True, origin=MemoryOrigin.USER_MESSAGE)
    assert not should_auto_pin(category="identity", fallback=False, origin=MemoryOrigin.USER_MESSAGE)
    assert not should_auto_pin(category="identity", fallback=True, origin=MemoryOrigin.ASSISTANT_OUTPUT)
    assert not should_auto_pin(category="preference", fallback=True, origin=MemoryOrigin.USER_MESSAGE)


def test_automatic_memory_audit_is_disabled_by_default():
    from services.memory import memory_extractor

    assert memory_extractor.AUTO_AUDIT_ENABLED is False

