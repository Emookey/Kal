"""Origin-aware admission policy for automatic personal-memory extraction."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any


class MemoryOrigin(str, Enum):
    USER_MESSAGE = "user_message"
    ASSISTANT_OUTPUT = "assistant_output"
    SHARED_KNOWLEDGE = "shared_knowledge"
    BUSINESS_DATA = "business_data"
    SYSTEM = "system"


_STRUCTURED_FIELD_RE = re.compile(
    r"\b(?:Record_ID|Client_ID|Device_ID|Primary_Device|User_ID|Printer_ID|"
    r"Ticket_ID|SOP_ID|Resource_ID|Assigned_User|RAM_GB|Below_16GB_RAM|"
    r"Storage_Type|Uses_HDD|Under_Spec_Device|Performance_Risk|"
    r"Upgrade_Priority|Operational_Criticality|Applies_To|Trigger)\b",
    re.IGNORECASE,
)
_ATOMIC_MARKER_RE = re.compile(r"\b(?:[A-Z][A-Z0-9_]*(?:_RECORD|_RULE)|END_[A-Z0-9_]+)\b")
_BUSINESS_ID_RE = re.compile(
    r"\b(?:ACME(?:-[A-Z0-9]+)+|PRT-(?:BOR|PD)(?:-[A-Z0-9]+)+)\b",
    re.IGNORECASE,
)


def automatic_extraction_messages(messages: list[Any]) -> list[Any]:
    """Keep only user-authored messages as automatic-memory evidence."""

    kept = []
    for message in messages or []:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if str(role or "").lower() == "user":
            kept.append(message)
    return kept


def is_business_or_rag_fact(text: str, *, origin: MemoryOrigin = MemoryOrigin.USER_MESSAGE) -> bool:
    """Reject structured/business evidence from the private-memory lane.

    The origin check is intentionally generic: future Beepy schemas do not
    need to be encoded here to keep non-user domains out of personal memory.
    Identifier/field patterns retain deployed GoodWill compatibility for
    user-pasted structured records.
    """

    if origin is not MemoryOrigin.USER_MESSAGE:
        return True
    value = text or ""
    return bool(
        _STRUCTURED_FIELD_RE.search(value)
        or _ATOMIC_MARKER_RE.search(value)
        or _BUSINESS_ID_RE.search(value)
    )


def should_auto_pin(*, category: str, fallback: bool, origin: MemoryOrigin) -> bool:
    return category == "identity" and fallback and origin is MemoryOrigin.USER_MESSAGE

