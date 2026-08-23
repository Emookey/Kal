"""Small explicit reconciliation seam between authoritative Library rows and RAG."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class RAGSyncStatus:
    source_id: str
    content_hash: str
    state: str
    indexed_count: int = 0
    detail: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


_LOCK = threading.RLock()
_STATUS: dict[str, RAGSyncStatus] = {}


def content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def record_status(
    source_id: str,
    content: str,
    state: str,
    *,
    indexed_count: int = 0,
    detail: Optional[str] = None,
) -> RAGSyncStatus:
    if state not in {"pending", "indexed", "degraded", "removed"}:
        raise ValueError(f"Unknown RAG sync state: {state}")
    status = RAGSyncStatus(source_id, content_hash(content), state, indexed_count, detail)
    with _LOCK:
        _STATUS[source_id] = status
    return status


def get_status(source_id: str) -> Optional[RAGSyncStatus]:
    with _LOCK:
        return _STATUS.get(source_id)


def clear_statuses_for_tests() -> None:
    with _LOCK:
        _STATUS.clear()
