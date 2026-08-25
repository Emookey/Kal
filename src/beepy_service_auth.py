"""Pure authentication helpers for the scoped Beepy service identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.retrieval_policy import (
    has_beepy_service_markers,
    is_exact_beepy_service_identity,
)


_BEEPY_SERVICE_ROUTES = {
    ("GET", "/api/integrations/beepy/capabilities"),
    ("POST", "/api/integrations/beepy/technical-chat"),
}


@dataclass(frozen=True)
class ApiTokenIdentity:
    owner: str
    scopes: list[str]
    profile: str | None
    beepy_service: bool = False


def beepy_service_route_allowed(method: str, path: str) -> bool:
    normalized_path = path.rstrip("/") or "/"
    return (str(method or "").upper(), normalized_path) in _BEEPY_SERVICE_ROUTES


def resolve_api_token_identity(
    owner: Any,
    scopes_value: Any,
    profile: Any,
    normalize_human_owner: Callable[[Any], str | None],
) -> ApiTokenIdentity | None:
    """Resolve a DB token row without treating a service as a human account."""

    scopes = [
        part.strip()
        for part in str(scopes_value or "chat").split(",")
        if part.strip()
    ]
    profile_value = str(profile).strip() if isinstance(profile, str) and profile.strip() else None
    if has_beepy_service_markers(owner, scopes, profile_value):
        if not is_exact_beepy_service_identity(owner, scopes, profile_value):
            return None
        return ApiTokenIdentity(str(owner).strip(), scopes, profile_value, True)

    human_owner = normalize_human_owner(owner)
    if not human_owner:
        return None
    return ApiTokenIdentity(human_owner, scopes, profile_value, False)
