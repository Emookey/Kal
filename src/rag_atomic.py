"""Bounded parser for atomic ``*_RECORD`` and ``*_RULE`` RAG documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Mapping


PARSER_VERSION = "atomic-rag-v1"
MAX_INPUT_CHARS = 2_000_000
MAX_RECORDS = 5_000
MAX_RECORD_CHARS = 100_000

_START_RE = re.compile(r"^[A-Z][A-Z0-9_]*(?:_RECORD|_RULE)$")

FIELD_METADATA: Mapping[str, str] = {
    "Record_ID": "record_id",
    "Client_ID": "client_id",
    "Device_ID": "device_id",
    "Primary_Device": "device_id",
    "User_ID": "user_id",
    "Printer_ID": "printer_id",
    "Ticket_ID": "ticket_id",
    "SOP_ID": "sop_id",
    "Resource_ID": "resource_id",
    "Assigned_User": "assigned_user",
    "User": "user_name",
    "Role": "role",
    "Client_Name": "client_name",
    "Application": "application_name",
    "RAM_GB": "ram_gb",
    "Below_16GB_RAM": "below_16gb_ram",
    "Storage_Type": "storage_type",
    "Uses_HDD": "uses_hdd",
    "Under_Spec_Device": "under_spec_device",
    "Performance_Risk": "performance_risk",
    "Upgrade_Priority": "upgrade_priority",
    "Operational_Criticality": "operational_criticality",
    "Applies_To": "applies_to",
    "Trigger": "trigger",
}


@dataclass(frozen=True)
class AtomicRecord:
    text: str
    metadata: dict


class AtomicRAGParseError(ValueError):
    pass


def _stable_record_id(record_type: str, text: str) -> str:
    digest = hashlib.sha256(f"{record_type}\0{text}".encode("utf-8")).hexdigest()[:20]
    return f"ATOMIC_{digest.upper()}"


def parse_atomic_rag_records(content: str) -> list[AtomicRecord]:
    """Parse complete atomic blocks deterministically or fail the document.

    Ordinary prose outside blocks is ignored for deployed grammar
    compatibility.  A recognized start marker must have its matching end
    marker; malformed or over-limit input never yields a partial index.
    """

    if not isinstance(content, str):
        raise AtomicRAGParseError("Atomic RAG content must be text")
    if len(content) > MAX_INPUT_CHARS:
        raise AtomicRAGParseError(f"Atomic RAG content exceeds {MAX_INPUT_CHARS} characters")

    lines = content.splitlines()
    records: list[AtomicRecord] = []
    seen_ids: set[str] = set()
    index = 0

    while index < len(lines):
        record_type = lines[index].strip()
        if not _START_RE.fullmatch(record_type):
            index += 1
            continue

        end_marker = f"END_{record_type}"
        block = [lines[index]]
        index += 1
        while index < len(lines) and lines[index].strip() != end_marker:
            nested = lines[index].strip()
            if _START_RE.fullmatch(nested):
                raise AtomicRAGParseError(
                    f"Atomic RAG record {record_type} encountered nested marker {nested}"
                )
            block.append(lines[index])
            if sum(len(row) + 1 for row in block) > MAX_RECORD_CHARS:
                raise AtomicRAGParseError(
                    f"Atomic RAG record {record_type} exceeds {MAX_RECORD_CHARS} characters"
                )
            index += 1

        if index >= len(lines):
            raise AtomicRAGParseError(
                f"Atomic RAG record {record_type} is missing {end_marker}"
            )
        block.append(lines[index])
        text = "\n".join(block).strip()

        fields: dict[str, str] = {}
        for row in block[1:-1]:
            if ": " not in row:
                continue
            key, value = row.split(": ", 1)
            fields[key.strip()] = value.strip()

        metadata = {"atomic": "true", "record_type": record_type, "parser_version": PARSER_VERSION}
        for source_key, metadata_key in FIELD_METADATA.items():
            value = fields.get(source_key)
            if value:
                metadata[metadata_key] = value

        metadata.setdefault("record_id", _stable_record_id(record_type, text))
        metadata["content_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        stable_id = str(metadata["record_id"]).upper()
        if stable_id in seen_ids:
            raise AtomicRAGParseError(f"Duplicate atomic Record_ID: {metadata['record_id']}")
        seen_ids.add(stable_id)
        records.append(AtomicRecord(text=text, metadata=metadata))
        if len(records) > MAX_RECORDS:
            raise AtomicRAGParseError(f"Atomic RAG document exceeds {MAX_RECORDS} records")
        index += 1

    return records

