"""Stateless shared-knowledge retrieval for the Beepy service contract."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from src.prompt_security import UNTRUSTED_CONTEXT_POLICY, untrusted_context_message
from src.retrieval_policy import RetrievalContext, beepy_service_principal
from src.structured_rag import retrieve_structured


logger = logging.getLogger(__name__)

MAX_SERVICE_SOURCES = 5
MAX_SOURCE_LABEL = 120
MAX_GROUNDING_LABEL = 40
MAX_EVIDENCE_RECORDS = 8
MAX_EVIDENCE_CHARS = 10_000
_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9 _.,:@()#&+\-]")


@dataclass(frozen=True)
class BeepyGrounding:
    status: str
    messages: list[dict[str, str]]
    sources: list[dict[str, str]]

    @property
    def used_shared_knowledge(self) -> bool:
        return self.status == "grounded"


def _bounded_text(value: Any, limit: int, fallback: str) -> str:
    text = " ".join(str(value or "").split())
    text = _SAFE_LABEL_RE.sub(" ", text)
    text = " ".join(text.split()).strip(" .")
    return (text[:limit].rstrip() or fallback)


def _safe_source_label(metadata: dict[str, Any]) -> str:
    """Return a bounded label without exposing a filesystem or URL path."""

    raw = metadata.get("title") or metadata.get("filename") or metadata.get("source")
    text = str(raw or "")
    parsed = urlparse(text)
    if parsed.scheme or parsed.netloc:
        text = (parsed.path or "").rstrip("/").split("/")[-1] or parsed.netloc
    else:
        text = text.replace("\\", "/").rstrip("/").split("/")[-1]
    return _bounded_text(text, MAX_SOURCE_LABEL, "shared knowledge")


def _safe_source(result: dict[str, Any]) -> dict[str, str]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    return {
        "label": _safe_source_label(metadata),
        "grounding": _bounded_text(
            metadata.get("grounding"), MAX_GROUNDING_LABEL, "semantic_support"
        ),
    }


def retrieve_beepy_shared_knowledge(rag_manager: Any, message: str) -> BeepyGrounding:
    """Retrieve only records allowed by Kal's service retrieval context.

    The function intentionally has no session, memory, upload, web, research,
    tool, or webhook dependency. Retrieval failures become a bounded degraded
    result instead of leaking datastore details.
    """

    base_messages = [
        {"role": "system", "content": UNTRUSTED_CONTEXT_POLICY},
        {
            "role": "system",
            "content": (
                "Answer the technical question directly. Use retrieved shared "
                "knowledge when supplied. Do not claim access to private Kal "
                "data, personal memory, tools, web search, or business systems."
            ),
        },
    ]
    if rag_manager is None:
        return BeepyGrounding("degraded", [*base_messages, {"role": "user", "content": message}], [])

    context = RetrievalContext.service(beepy_service_principal())
    try:
        semantic_results = rag_manager.search(
            message,
            k=10,
            owner=beepy_service_principal(),
            access_context=context,
        )
        scan = getattr(rag_manager, "scan_accessible", None)
        if not callable(scan):
            raise RuntimeError("shared retrieval scan is unavailable")
        authorized_records = scan(access_context=context)
        outcome = retrieve_structured(
            message,
            authorized_records,
            semantic_results,
            history=[],
            semantic_threshold=0.35,
        )
        relevant = [
            result
            for result in (outcome.results or [])
            if isinstance(result, dict)
            and context.allows_metadata(result.get("metadata") or {})
            and str(result.get("document") or "").strip()
        ][:MAX_EVIDENCE_RECORDS]
    except Exception as exc:
        logger.warning("Beepy shared retrieval degraded (%s)", type(exc).__name__)
        return BeepyGrounding("degraded", [*base_messages, {"role": "user", "content": message}], [])

    if not relevant:
        return BeepyGrounding("no_match", [*base_messages, {"role": "user", "content": message}], [])

    evidence_parts = []
    sources = []
    seen_sources = set()
    for result in relevant:
        safe_source = _safe_source(result)
        source_key = (safe_source["label"], safe_source["grounding"])
        if source_key not in seen_sources and len(sources) < MAX_SERVICE_SOURCES:
            seen_sources.add(source_key)
            sources.append(safe_source)
        document = str(result.get("document") or "").strip()
        if document:
            evidence_parts.append(
                f"[{safe_source['label']} | {safe_source['grounding']}]\n{document}"
            )

    evidence = "\n\n---\n\n".join(evidence_parts)[:MAX_EVIDENCE_CHARS]
    messages = [*base_messages]
    if evidence:
        messages.append(untrusted_context_message("approved shared technical knowledge", evidence))
    messages.append({"role": "user", "content": message})
    return BeepyGrounding("grounded", messages, sources)
