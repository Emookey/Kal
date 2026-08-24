"""Central access policy for RAG and future retrieval domains.

The vector index is a derived store, so every read is authorized again from
metadata.  In particular, an omitted owner is never interpreted as a request
for every document.  Broad maintenance reads require an explicit system
context.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional


LEGACY_SHARED_PRINCIPAL = "__goodwill_shared__"
SHARED_PRINCIPAL_ENV = "ODYSSEUS_SHARED_RAG_PRINCIPAL"
BEEPY_SERVICE_PRINCIPAL_ENV = "ODYSSEUS_BEEPY_SERVICE_PRINCIPAL"
BEEPY_SHARED_RAG_SCOPE = "beepy:shared-rag"


def shared_principal() -> str:
    """Return the configured shared-knowledge principal.

    The legacy value remains the default until the later Odysseus-to-Kal
    identity migration.  Keeping the name behind this function avoids
    scattering that compatibility identifier through retrieval code.
    """

    value = (os.getenv(SHARED_PRINCIPAL_ENV) or LEGACY_SHARED_PRINCIPAL).strip()
    return value or LEGACY_SHARED_PRINCIPAL


def beepy_service_principal() -> str:
    value = (os.getenv(BEEPY_SERVICE_PRINCIPAL_ENV) or "service:beepy").strip()
    return value or "service:beepy"


def is_beepy_service_request(request: Any) -> bool:
    state = getattr(request, "state", None)
    return bool(
        getattr(state, "api_token", False)
        and BEEPY_SHARED_RAG_SCOPE in set(getattr(state, "api_token_scopes", []) or [])
    )


class PrincipalKind(str, Enum):
    ANONYMOUS = "anonymous"
    USER = "user"
    SERVICE = "service"
    SYSTEM = "system"


@dataclass(frozen=True)
class RetrievalContext:
    """Identity and domain grants for RAG read and mutation operations.

    Read and delete authorization are deliberately separate.  A user may read
    approved shared knowledge, but that grant never conveys authority to
    mutate it.
    """

    kind: PrincipalKind
    principal: Optional[str] = None
    allow_private: bool = False
    allow_shared: bool = False
    allow_unowned_legacy: bool = False
    allow_all: bool = False

    @classmethod
    def user(cls, owner: str) -> "RetrievalContext":
        owner = (owner or "").strip()
        if not owner:
            raise ValueError("Authenticated retrieval requires an explicit owner")
        return cls(PrincipalKind.USER, owner, allow_private=True, allow_shared=True)

    @classmethod
    def service(cls, principal: str = "beepy") -> "RetrievalContext":
        principal = (principal or "").strip()
        if not principal:
            raise ValueError("Service retrieval requires an explicit principal")
        return cls(PrincipalKind.SERVICE, principal, allow_shared=True)

    @classmethod
    def anonymous(cls) -> "RetrievalContext":
        # Compatibility for explicitly unauthenticated/single-user content.
        # This is still fail-closed for private and shared-owner rows.
        return cls(PrincipalKind.ANONYMOUS, allow_unowned_legacy=True)

    @classmethod
    def system(cls, principal: str = "odysseus-system") -> "RetrievalContext":
        principal = (principal or "").strip()
        if not principal:
            raise ValueError("System retrieval requires an explicit principal")
        return cls(PrincipalKind.SYSTEM, principal, allow_all=True)

    def allows_owner(self, owner: Any) -> bool:
        if self.allow_all:
            return True
        normalized = str(owner).strip() if owner is not None else ""
        if not normalized:
            return self.allow_unowned_legacy
        if self.allow_shared and normalized == shared_principal():
            return True
        return bool(self.allow_private and self.principal and normalized == self.principal)

    def allows_metadata(self, metadata: Mapping[str, Any] | None) -> bool:
        return self.allows_owner((metadata or {}).get("owner"))

    def allows_delete_metadata(self, metadata: Mapping[str, Any] | None) -> bool:
        """Authorize deletion without reusing the broader read policy.

        Users may delete only rows owned exactly by their own principal.
        Anonymous and service contexts cannot delete RAG records.  Broad
        deletion is reserved for an explicitly constructed system context.
        """

        if self.kind is PrincipalKind.SYSTEM:
            return self.allow_all
        if self.kind is not PrincipalKind.USER or not self.principal:
            return False
        owner = (metadata or {}).get("owner")
        normalized = str(owner).strip() if owner is not None else ""
        return bool(
            normalized
            and normalized != shared_principal()
            and normalized == self.principal
        )

    def chroma_where(self) -> Optional[dict]:
        """Return a narrowing Chroma filter when it can express the policy.

        Callers must still apply :meth:`allows_metadata` to returned rows.
        Anonymous legacy rows may have a missing ``owner`` key, which is not
        portably expressible across supported Chroma versions, so that case is
        post-filtered.
        """

        if self.allow_all or self.allow_unowned_legacy:
            return None
        owners = []
        if self.allow_private and self.principal:
            owners.append(self.principal)
        if self.allow_shared:
            owners.append(shared_principal())
        if not owners:
            # A deliberately impossible value keeps the datastore query closed.
            return {"owner": "__odysseus_no_access__"}
        if len(owners) == 1:
            return {"owner": owners[0]}
        return {"$or": [{"owner": value} for value in owners]}


def coerce_retrieval_context(
    *, owner: Optional[str] = None, access_context: Optional[RetrievalContext] = None
) -> RetrievalContext:
    """Adapt legacy ``owner=`` callers without restoring null-owner bypasses."""

    if access_context is not None:
        if not isinstance(access_context, RetrievalContext):
            raise TypeError("access_context must be a RetrievalContext")
        return access_context
    if owner:
        return RetrievalContext.user(owner)
    return RetrievalContext.anonymous()
