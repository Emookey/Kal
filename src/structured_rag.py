"""Deterministic structured retrieval helpers for atomic MSP knowledge.

This module contains no model/provider calls and no datastore assumptions.  It
operates on already-authorized rows supplied by the centralized retrieval
policy, which keeps exact, aggregate, entity, and SOP behavior independently
testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


IDENTIFIER_RE = re.compile(r"\b[A-Za-z0-9]{2,}(?:-[A-Za-z0-9]{2,}){2,}\b")
REFERENCE_RE = re.compile(
    r"\b(?:it|them|those|these|both|each|either|neither|former|latter)\b|"
    r"\b(?:first one|second one|which one|which of|other one|other device|"
    r"those two|these two|that user|this user|that device|this device|"
    r"weaker system|stronger system)\b",
    re.IGNORECASE,
)
ID_METADATA_FIELDS = (
    "record_id",
    "client_id",
    "device_id",
    "user_id",
    "printer_id",
    "ticket_id",
    "sop_id",
    "resource_id",
)
SOP_PRIORITY = {
    "SOP_TRIGGER_RECORD": 0,
    "SOP_ACTION_RECORD": 1,
    "SOP_GUARDRAIL_RECORD": 2,
    "SOP_DECISION_RECORD": 3,
    "SOP_OUTPUT_RECORD": 4,
    "ESCALATION_RULE": 5,
    "PRIORITY_RULE": 6,
}
SEMANTIC_SUPPORT_TERMS = (
    "troubleshoot", "troubleshooting", "procedure", "sop", "outage",
    "error", "failed", "fails", "frozen", "security", "mfa",
)


@dataclass(frozen=True)
class AggregateQuery:
    field: str
    label: str
    operator: str
    value: str


@dataclass(frozen=True)
class StructuredOutcome:
    results: list[dict]
    identifiers: list[str]
    explicit_identifiers: list[str]
    carried_identifiers: list[str]
    missing_identifiers: list[str]
    aggregate: AggregateQuery | None = None


def extract_identifiers(text: str) -> list[str]:
    identifiers = []
    for candidate in IDENTIFIER_RE.findall(text or ""):
        if not any(char.isdigit() for char in candidate):
            continue
        normalized = candidate.upper()
        if normalized not in identifiers:
            identifiers.append(normalized)
    return identifiers


def _message_role_and_text(message: Any) -> tuple[str, str]:
    if isinstance(message, Mapping):
        role, content = message.get("role"), message.get("content", "")
    else:
        role, content = getattr(message, "role", None), getattr(message, "content", "")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        content = "\n".join(parts)
    return str(role or ""), content if isinstance(content, str) else ""


def carry_recent_user_identifiers(
    message: str, history: Sequence[Any] | None, explicit: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Carry at most two active IDs, and only from recent user turns."""

    lowered = (message or "").lower()
    has_reference = bool(REFERENCE_RE.search(lowered))
    if not has_reference and "which" in lowered:
        has_reference = any(
            word in lowered
            for word in ("more", "less", "higher", "lower", "faster", "slower", "weaker", "stronger", "better", "worse")
        )
    active = list(explicit)
    carried: list[str] = []
    if not has_reference:
        return active, carried

    for entry in reversed(list(history or [])[-30:]):
        role, content = _message_role_and_text(entry)
        if role != "user" or content.strip() == (message or "").strip():
            continue
        for identifier in extract_identifiers(content):
            if identifier not in active:
                active.append(identifier)
                carried.append(identifier)
            if len(active) >= 2:
                return active, carried
    return active, carried


def _is_atomic(result: Mapping[str, Any]) -> bool:
    return str((result.get("metadata") or {}).get("atomic", "")).lower() == "true"


def _record_key(result: Mapping[str, Any]) -> tuple:
    metadata = result.get("metadata") or {}
    return metadata.get("source"), metadata.get("chunk_id"), result.get("document", "")


def dedupe_records(records: Iterable[Mapping[str, Any]]) -> list[dict]:
    seen = set()
    output = []
    for record in records:
        key = _record_key(record)
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(record))
    return output


def _contains_exact(text: str, value: str) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])",
            text or "",
            flags=re.IGNORECASE,
        )
    )


def _authoritative(record: Mapping[str, Any], *, score: float = 1.0) -> dict:
    result = dict(record)
    metadata = dict(result.get("metadata") or {})
    metadata["grounding"] = "authoritative_structured"
    result["metadata"] = metadata
    result["similarity"] = score
    result["vector_similarity"] = score
    result["keyword_score"] = score
    return result


def resolve_entities(message: str, atomic_records: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[dict], list[str]]:
    device_ids: list[str] = []
    application_records: list[dict] = []
    people: list[str] = []
    for record in atomic_records:
        metadata = record.get("metadata") or {}
        record_type = metadata.get("record_type")
        if record_type in {"USER_ASSIGNMENT_RECORD", "DEVICE_ASSIGNMENT_RECORD"}:
            person = metadata.get("assigned_user") or metadata.get("user_name")
            device_id = metadata.get("device_id")
            if person and device_id and _contains_exact(message, str(person)):
                normalized = str(device_id).upper()
                if normalized not in device_ids:
                    device_ids.append(normalized)
                if str(person) not in people:
                    people.append(str(person))
        application = metadata.get("application_name")
        if application and _contains_exact(message, str(application)):
            application_records.append(_authoritative(record))
    return device_ids, dedupe_records(application_records), people


def preferred_record_types(message: str, *, resolved_people: bool = False, application: bool = False) -> tuple[list[str], bool]:
    lowered = (message or "").lower()
    preferred: list[str] = []

    def add(*values: str) -> None:
        for value in values:
            if value not in preferred:
                preferred.append(value)

    if any(term in lowered for term in ("assigned user", "assigned_user", "who uses", "who is assigned", "role", "client")):
        add("USER_ASSIGNMENT_RECORD", "DEVICE_ASSIGNMENT_RECORD")
    if any(term in lowered for term in ("ram", "storage", "cpu", "operating system", " os ", "hardware")):
        add("DEVICE_HARDWARE_RECORD")
    if any(term in lowered for term in ("risk", "upgrade", "under spec", "under-spec", "below 16", "below_16")):
        add("DEVICE_RISK_RECORD", "DEVICE_HARDWARE_RECORD")
    if application or any(term in lowered for term in ("application", "applications", "software", "apps")):
        add("DEVICE_APPLICATION_RECORD", "APPLICATION_RECORD")
    if any(term in lowered for term in ("known issue", "known issues", "recurring issue")):
        add("DEVICE_KNOWN_ISSUE_RECORD")
    if any(term in lowered for term in ("troubleshoot", "troubleshooting", "what should i check", "check first", "direct hint")):
        add("DEVICE_DIRECT_HINT_RECORD", "DEVICE_KNOWN_ISSUE_RECORD")
    if "printer" in lowered or "printing" in lowered:
        add("PRINTER_CONFIGURATION_RECORD", "PRINTER_USAGE_RECORD", "PRINTER_DIRECT_HINT_RECORD", "PRINTER_KNOWN_ISSUE_RECORD")
    if resolved_people:
        add("USER_ASSIGNMENT_RECORD", "DEVICE_ASSIGNMENT_RECORD")
    explicit = bool(preferred)
    if not preferred:
        add("USER_ASSIGNMENT_RECORD", "DEVICE_ASSIGNMENT_RECORD", "DEVICE_HARDWARE_RECORD", "DEVICE_RISK_RECORD", "DEVICE_DIRECT_HINT_RECORD")
    return preferred, explicit


def exact_lookup(
    identifiers: Sequence[str],
    atomic_records: Sequence[Mapping[str, Any]],
    preferred: Sequence[str],
    *,
    filter_to_preferred: bool,
    limit: int = 10,
) -> tuple[list[dict], list[str]]:
    by_id: dict[str, list[dict]] = {identifier: [] for identifier in identifiers}
    for record in atomic_records:
        metadata = record.get("metadata") or {}
        metadata_ids = {
            str(metadata[field]).upper()
            for field in ID_METADATA_FIELDS
            if metadata.get(field)
        }
        document = str(record.get("document") or "")
        for identifier in identifiers:
            if identifier in metadata_ids or _contains_exact(document, identifier):
                by_id[identifier].append(_authoritative(record))

    missing = [identifier for identifier, records in by_id.items() if not records]
    for identifier in missing:
        document = (
            "EXACT_ID_NOT_FOUND_RECORD\n"
            f"Identifier: {identifier}\n"
            "Status: Not_Found\n"
            "Instruction: No authoritative atomic record exists for this exact identifier. "
            "Do not substitute, guess, or use a similar identifier.\n"
            "END_EXACT_ID_NOT_FOUND_RECORD"
        )
        by_id[identifier] = [_authoritative({
            "id": f"not-found:{identifier}",
            "document": document,
            "metadata": {
                "source": f"synthetic:not-found:{identifier}",
                "filename": "Structured-Router",
                "chunk_id": 0,
                "atomic": "true",
                "record_type": "EXACT_ID_NOT_FOUND_RECORD",
                "record_id": f"NOT_FOUND_{identifier}",
            },
        })]

    rank = {record_type: index for index, record_type in enumerate(preferred)}
    for identifier in identifiers:
        candidates = by_id[identifier]
        if filter_to_preferred:
            selected = [
                record for record in candidates
                if (record.get("metadata") or {}).get("record_type") in rank
                or (record.get("metadata") or {}).get("record_type") == "EXACT_ID_NOT_FOUND_RECORD"
            ]
            if selected:
                candidates = selected
        candidates.sort(key=lambda record: (
            rank.get((record.get("metadata") or {}).get("record_type"), len(rank) + 100),
            str((record.get("metadata") or {}).get("record_type", "")),
            str((record.get("metadata") or {}).get("record_id", "")),
        ))
        by_id[identifier] = candidates

    # Fair round-robin: every ID receives one result before any receives two.
    selected: list[dict] = []
    round_number = 0
    while len(selected) < limit:
        added = False
        for identifier in identifiers:
            candidates = by_id[identifier]
            if round_number < len(candidates):
                selected.append(candidates[round_number])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        round_number += 1
    return selected, missing


def parse_aggregate_query(message: str) -> AggregateQuery | None:
    lowered = (message or "").lower()
    if not re.search(r"\b(?:all|every|list|which)\b", lowered):
        return None
    if not re.search(r"\b(?:device|devices|workstation|workstations|endpoint|endpoints)\b", lowered):
        return None
    for name in ("below_16gb_ram", "uses_hdd", "under_spec_device"):
        match = re.search(rf"\b{re.escape(name)}\b\s*(?:is|=|equals|set to)?\s*(yes|no)\b", lowered)
        if match:
            return AggregateQuery(name, name, "eq", match.group(1))
    natural_boolean = (
        (r"\b(?:below|under)\s*16\s*gb\b", "below_16gb_ram"),
        (r"\b(?:uses?|using|with)\s+(?:an?\s+)?hdd\b", "uses_hdd"),
        (r"\bunder[- ]spec(?:ification)?\b", "under_spec_device"),
    )
    for pattern, field in natural_boolean:
        if field == "below_16gb_ram" and re.search(
            r"(?:ram|ram_gb).*?(?:less than|under|below)\s*\d+", lowered
        ):
            continue
        if re.search(pattern, lowered):
            return AggregateQuery(field, field, "eq", "yes")
    for name in ("performance_risk", "upgrade_priority", "operational_criticality", "storage_type"):
        match = re.search(
            rf"\b{re.escape(name)}\b\s*(?:is|=|equals|set to)?\s*(low|medium|high|critical|ssd|hdd)\b",
            lowered,
        )
        if match:
            return AggregateQuery(name, name, "eq", match.group(1))
    patterns = (
        (r"(?:ram|ram_gb).*?(?:less than|under|below)\s*(\d+)\s*gb", "lt"),
        (r"(?:ram|ram_gb).*?(?:more than|over)\s*(\d+)\s*gb", "gt"),
        (r"(?:ram|ram_gb).*?(?:at least)\s*(\d+)\s*gb", "gte"),
        (r"(?:ram|ram_gb).*?(?:at most)\s*(\d+)\s*gb", "lte"),
    )
    for pattern, operator in patterns:
        match = re.search(pattern, lowered)
        if match:
            return AggregateQuery("ram_gb", "RAM_GB", operator, match.group(1))
    return None


def _aggregate_matches(value: str, query: AggregateQuery) -> bool:
    if query.operator == "eq":
        return str(value).strip().lower() == query.value.strip().lower()
    try:
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        number = float(match.group()) if match else None
        target = float(query.value)
    except (TypeError, ValueError):
        return False
    if number is None:
        return False
    return {
        "lt": number < target,
        "gt": number > target,
        "lte": number <= target,
        "gte": number >= target,
    }.get(query.operator, False)


def build_aggregate_result(query: AggregateQuery, atomic_records: Sequence[Mapping[str, Any]]) -> dict:
    devices: dict[str, dict] = {}
    for record in atomic_records:
        metadata = record.get("metadata") or {}
        device_id, raw_value = metadata.get("device_id"), metadata.get(query.field)
        if not device_id or raw_value is None:
            continue
        entry = devices.setdefault(str(device_id), {"client_id": metadata.get("client_id"), "values": set(), "sources": set()})
        entry["values"].add(str(raw_value))
        if metadata.get("filename"):
            entry["sources"].add(str(metadata["filename"]))

    matches, conflicts = [], []
    for device_id, info in devices.items():
        values = sorted(info["values"])
        if len(values) > 1:
            conflicts.append((device_id, values))
        elif values and _aggregate_matches(values[0], query):
            matches.append((str(info.get("client_id") or "Unknown"), device_id, values[0]))
    matches.sort()
    conflicts.sort()
    lines = [
        "AGGREGATE_QUERY_RESULT_RECORD",
        "Authority: Authoritative_Structured_Result",
        "Operation_Type: Read_Only_Query",
        "Computation_Status: Complete",
        "Access_Scope: Results contain only records authorized before aggregation.",
        f"Query_Field: {query.label}",
        f"Operator: {query.operator}",
        f"Query_Value: {query.value}",
        f"Match_Count: {len(matches)}",
        "Matches:",
    ]
    lines.extend(
        f"- Client_ID={client_id} | Device_ID={device_id} | {query.label}={value}"
        for client_id, device_id, value in matches
    )
    if conflicts:
        lines.append("Conflicts:")
        lines.extend(f"- Device_ID={device_id} | Values={values}" for device_id, values in conflicts)
    lines.extend([
        "Instruction: Present every matching device exactly once. Do not add devices not listed.",
        "END_AGGREGATE_QUERY_RESULT_RECORD",
    ])
    return _authoritative({
        "id": f"aggregate:{query.field}:{query.operator}:{query.value}",
        "document": "\n".join(lines),
        "metadata": {
            "source": "synthetic:aggregate",
            "filename": "Structured-Aggregate",
            "chunk_id": 0,
            "atomic": "true",
            "record_type": "AGGREGATE_QUERY_RESULT_RECORD",
        },
    })


def assemble_sops(message: str, atomic_records: Sequence[Mapping[str, Any]], client_ids: set[str]) -> list[dict]:
    tokens = {
        token for token in re.findall(r"[a-z0-9]+", (message or "").lower())
        if len(token) >= 3 and token not in {
            "the", "and", "for", "with", "from", "this", "that", "what", "which", "who", "how", "should", "would", "could", "can", "our", "you",
        }
    }
    groups: dict[str, list[tuple[float, dict]]] = {}
    for record in atomic_records:
        metadata = record.get("metadata") or {}
        record_type = str(metadata.get("record_type") or "")
        if not (record_type.startswith("SOP_") or record_type in {"ESCALATION_RULE", "PRIORITY_RULE"}):
            continue
        scoped = str(metadata.get("client_id") or "")
        applies_to = str(metadata.get("applies_to") or "")
        if client_ids and (scoped or applies_to):
            if scoped not in client_ids and not any(client_id in applies_to for client_id in client_ids):
                continue
        overlap = tokens & set(re.findall(r"[a-z0-9]+", str(record.get("document") or "").lower()))
        score = len(overlap) / max(len(tokens), 1)
        if overlap and record_type == "SOP_TRIGGER_RECORD":
            score += 0.15
        sop_id = str(metadata.get("sop_id") or f"NO_SOP_ID:{metadata.get('source')}")
        groups.setdefault(sop_id, []).append((score, _authoritative(record, score=1.0)))
    groups = {
        sop_id: entries
        for sop_id, entries in groups.items()
        if any(score > 0 for score, _ in entries)
    }
    if not groups:
        return []
    ranked = sorted(groups, key=lambda sop_id: (-max(score for score, _ in groups[sop_id]), sop_id))[:2]
    output = []
    for sop_id in ranked:
        # Complete assembly: all matching records for each selected SOP, not a
        # top-k subset that can silently omit decision/output/guardrail steps.
        records = [record for _, record in groups[sop_id]]
        records.sort(key=lambda record: (
            SOP_PRIORITY.get(str((record.get("metadata") or {}).get("record_type")), 99),
            str((record.get("metadata") or {}).get("record_id", "")),
        ))
        output.extend(records)
    return dedupe_records(output)


def retrieve_structured(
    message: str,
    authorized_records: Sequence[Mapping[str, Any]],
    semantic_results: Sequence[Mapping[str, Any]],
    *,
    history: Sequence[Any] | None = None,
    semantic_threshold: float = 0.35,
    limit: int = 20,
) -> StructuredOutcome:
    atomic = dedupe_records(record for record in authorized_records if _is_atomic(record))
    explicit = extract_identifiers(message)
    identifiers, carried = carry_recent_user_identifiers(message, history, explicit)
    entity_ids, application_records, people = resolve_entities(message, atomic)
    for identifier in entity_ids:
        if identifier not in identifiers:
            identifiers.append(identifier)

    preferred, explicit_preference = preferred_record_types(
        message, resolved_people=bool(people), application=bool(application_records)
    )
    exact, missing = exact_lookup(
        identifiers, atomic, preferred, filter_to_preferred=explicit_preference, limit=10
    ) if identifiers else ([], [])

    aggregate = parse_aggregate_query(message)
    if aggregate:
        return StructuredOutcome(
            [build_aggregate_result(aggregate, atomic)], identifiers, explicit, carried, missing, aggregate
        )

    client_ids = {
        str((record.get("metadata") or {}).get("client_id"))
        for record in exact
        if (record.get("metadata") or {}).get("client_id")
    }
    procedural = any(term in (message or "").lower() for term in SEMANTIC_SUPPORT_TERMS)
    sop_records = assemble_sops(message, atomic, client_ids) if procedural else []
    authoritative = dedupe_records([*exact, *application_records, *sop_records])

    # Exact not-found is an authoritative negative and suppresses semantic
    # substitution.  Aggregates above also suppress the semantic lane.
    include_semantic = not missing and (procedural or (not identifiers and not authoritative))
    semantic = []
    if include_semantic:
        for record in semantic_results:
            if float(record.get("similarity", 0) or 0) < semantic_threshold:
                continue
            item = dict(record)
            metadata = dict(item.get("metadata") or {})
            metadata.setdefault("grounding", "semantic_support")
            item["metadata"] = metadata
            semantic.append(item)

    return StructuredOutcome(
        dedupe_records([*authoritative, *semantic])[:limit],
        identifiers,
        explicit,
        carried,
        missing,
        None,
    )
