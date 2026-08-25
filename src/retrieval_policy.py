"""Central access policy for RAG and future retrieval domains.

The vector index is a derived store, so every read is authorized again from
metadata.  In particular, an omitted owner is never interpreted as a request
for every document.  Broad maintenance reads require an explicit system
context.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional


LEGACY_SHARED_PRINCIPAL = "__goodwill_shared__"
SHARED_PRINCIPAL_ENV = "KAL_SHARED_RAG_PRINCIPAL"
LEGACY_SHARED_PRINCIPAL_ENV = "ODYSSEUS_SHARED_RAG_PRINCIPAL"
BEEPY_SERVICE_PRINCIPAL_ENV = "KAL_BEEPY_SERVICE_PRINCIPAL"
LEGACY_BEEPY_SERVICE_PRINCIPAL_ENV = "ODYSSEUS_BEEPY_SERVICE_PRINCIPAL"
BEEPY_SHARED_RAG_SCOPE = "beepy:shared-rag"
BEEPY_SERVICE_TOKEN_PROFILE = "beepy_shared_rag"
BEEPY_SERVICE_SCOPES = ("chat", BEEPY_SHARED_RAG_SCOPE)
_PRINCIPAL_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class KalConfigurationError(RuntimeError):
    """Sanitized fail-closed error for conflicting integration settings."""


def _canonical_env_alias(canonical: str, legacy: str, default: str) -> str:
    """Resolve a canonical setting and its compatibility alias.

    Empty values are treated as unset. When both names are populated they
    must have the same stripped semantic value; the exception intentionally
    names only the settings, never their contents.
    """

    canonical_value = (os.getenv(canonical) or "").strip()
    legacy_value = (os.getenv(legacy) or "").strip()
    if canonical_value and legacy_value and canonical_value != legacy_value:
        raise KalConfigurationError(
            f"Conflicting configuration: {canonical} and {legacy} must match"
        )
    value = canonical_value or legacy_value or default
    if not _PRINCIPAL_RE.fullmatch(value):
        raise KalConfigurationError(f"Invalid configuration: {canonical}")
    return value


def shared_principal() -> str:
    """Return the configured shared-knowledge principal.

    The legacy value remains the default until the later Odysseus-to-Kal
    identity migration.  Keeping the name behind this function avoids
    scattering that compatibility identifier through retrieval code.
    """

    return _canonical_env_alias(
        SHARED_PRINCIPAL_ENV,
        LEGACY_SHARED_PRINCIPAL_ENV,
        LEGACY_SHARED_PRINCIPAL,
    )


def beepy_service_principal() -> str:
    value = _canonical_env_alias(
        BEEPY_SERVICE_PRINCIPAL_ENV,
        LEGACY_BEEPY_SERVICE_PRINCIPAL_ENV,
        "service:beepy",
    )
    if not value.startswith("service:"):
        raise KalConfigurationError(
            f"Invalid configuration: {BEEPY_SERVICE_PRINCIPAL_ENV}"
        )
    return value


def validate_beepy_service_configuration() -> tuple[str, str]:
    """Resolve and cross-check the service owner and shared-data owner."""

    service_owner = beepy_service_principal()
    shared_owner = shared_principal()
    if service_owner == shared_owner:
        raise KalConfigurationError(
            "Invalid configuration: KAL_BEEPY_SERVICE_PRINCIPAL and "
            "KAL_SHARED_RAG_PRINCIPAL must be distinct"
        )
    return service_owner, shared_owner


def has_beepy_service_markers(
    owner: Any, scopes: Any, profile: Any = None
) -> bool:
    """Return whether a token row claims any part of the Beepy identity."""

    service_owner, _ = validate_beepy_service_configuration()
    scope_values = tuple(scopes) if isinstance(scopes, (list, tuple)) else ()
    return bool(
        str(owner or "").strip() == service_owner
        or BEEPY_SHARED_RAG_SCOPE in scope_values
        or str(profile or "").strip() == BEEPY_SERVICE_TOKEN_PROFILE
    )


def is_exact_beepy_service_identity(
    owner: Any, scopes: Any, profile: Any
) -> bool:
    """Validate the immutable synthetic Beepy owner/profile/scope tuple."""

    if not isinstance(scopes, (list, tuple)):
        return False
    service_owner, _ = validate_beepy_service_configuration()
    return bool(
        str(owner or "").strip() == service_owner
        and str(profile or "").strip() == BEEPY_SERVICE_TOKEN_PROFILE
        and tuple(scopes) == BEEPY_SERVICE_SCOPES
    )


def is_beepy_service_request(request: Any) -> bool:
    state = getattr(request, "state", None)
    return bool(
        getattr(state, "api_token", False)
        and is_exact_beepy_service_identity(
            getattr(state, "api_token_owner", None),
            getattr(state, "api_token_scopes", None),
            getattr(state, "api_token_profile", None),
        )
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
