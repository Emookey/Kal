"""API Token management routes — /api/tokens/*."""

import secrets
import uuid

import bcrypt
from fastapi import APIRouter, HTTPException, Request, Form

from core.database import get_db_session, ApiToken
from core.middleware import require_admin
from src.auth_helpers import get_current_user
from src.retrieval_policy import (
    BEEPY_SERVICE_SCOPES,
    BEEPY_SERVICE_TOKEN_PROFILE,
    BEEPY_SHARED_RAG_SCOPE,
    has_beepy_service_markers,
    is_exact_beepy_service_identity,
    KalConfigurationError,
    validate_beepy_service_configuration,
)

MAX_NAME_LEN = 100
DEFAULT_SCOPES = "chat"
ALLOWED_SCOPES = {
    "chat",
    "todos:read",
    "todos:write",
    "documents:read",
    "documents:write",
    "email:read",
    "email:draft",
    "email:send",
    "calendar:read",
    "calendar:write",
    "memory:read",
    "memory:write",
    "cookbook:read",
    "cookbook:launch",
    BEEPY_SHARED_RAG_SCOPE,
}
TOKEN_PROFILES = {
    "chat": ["chat"],
    "codex_todos": ["todos:read", "todos:write"],
    "codex_documents": ["documents:read", "documents:write"],
    "codex_email_drafts": ["email:read", "email:draft", "documents:read", "documents:write"],
    # Deliberate service identity: chat transport plus shared RAG only.
    BEEPY_SERVICE_TOKEN_PROFILE: list(BEEPY_SERVICE_SCOPES),
}


def _normalize_scopes(scopes: str | list[str] | None = None, profile: str | None = None) -> list[str]:
    profile = profile if isinstance(profile, str) else None
    profile_key = (profile or "").strip()
    if profile_key:
        if profile_key not in TOKEN_PROFILES:
            raise HTTPException(400, "Unknown token profile")
        requested = list(TOKEN_PROFILES[profile_key])
    elif isinstance(scopes, list):
        requested = [str(s).strip() for s in scopes if str(s).strip()]
    elif isinstance(scopes, str) and scopes:
        requested = [s.strip() for s in scopes.replace(" ", ",").split(",") if s.strip()]
    else:
        requested = [DEFAULT_SCOPES]

    normalized = []
    for scope in requested:
        if scope not in ALLOWED_SCOPES:
            raise HTTPException(400, f"Unknown token scope: {scope}")
        if scope not in normalized:
            normalized.append(scope)

    def ensure_before(write_scope: str, read_scope: str):
        if write_scope not in normalized or read_scope in normalized:
            return
        idx = normalized.index(write_scope)
        normalized.insert(idx, read_scope)

    ensure_before("todos:write", "todos:read")
    ensure_before("documents:write", "documents:read")
    ensure_before("calendar:write", "calendar:read")
    ensure_before("memory:write", "memory:read")
    ensure_before("email:draft", "email:read")
    ensure_before("cookbook:launch", "cookbook:read")

    return normalized or [DEFAULT_SCOPES]


def _service_profile_scopes(
    scopes: str | list[str] | None, profile: str | None
) -> tuple[list[str], str | None]:
    """Normalize token input while locking the Beepy service profile."""

    profile_key = (profile or "").strip() if isinstance(profile, str) else ""
    explicit_scopes = scopes if isinstance(scopes, (str, list)) else None
    normalized = _normalize_scopes(explicit_scopes, profile_key or None)
    if profile_key == BEEPY_SERVICE_TOKEN_PROFILE:
        if explicit_scopes is not None:
            explicit = _normalize_scopes(explicit_scopes)
            if tuple(explicit) != BEEPY_SERVICE_SCOPES:
                raise HTTPException(400, "Beepy service profile scopes are immutable")
        return list(BEEPY_SERVICE_SCOPES), profile_key
    if BEEPY_SHARED_RAG_SCOPE in normalized:
        raise HTTPException(400, "Beepy shared RAG scope requires its service profile")
    return normalized, profile_key or None


def setup_api_token_routes() -> APIRouter:
    router = APIRouter(prefix="/api", tags=["api_tokens"])

    @router.get("/tokens")
    def list_tokens(request: Request):
        require_admin(request)
        with get_db_session() as db:
            tokens = db.query(ApiToken).all()
            return [
                {
                    "id": t.id,
                    "name": t.name,
                    "owner": getattr(t, "owner", None),
                    "token_prefix": t.token_prefix,
                    "profile": getattr(t, "profile", None),
                    "scopes": [s.strip() for s in (getattr(t, "scopes", "") or DEFAULT_SCOPES).split(",") if s.strip()],
                    "is_active": t.is_active,
                    "last_used_at": t.last_used_at.isoformat() if t.last_used_at else None,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                }
                for t in tokens
            ]

    def _invalidate_cache(request: Request):
        """Tell the auth middleware its cached token map is stale."""
        try:
            invalidator = getattr(request.app.state, "invalidate_token_cache", None)
            if invalidator:
                invalidator()
        except Exception:
            pass

    @router.get("/tokens/profiles")
    def token_profiles(request: Request):
        require_admin(request)
        return {
            "profiles": TOKEN_PROFILES,
            "allowed_scopes": sorted(ALLOWED_SCOPES),
        }

    @router.post("/tokens")
    def create_token(
        request: Request,
        name: str = Form(""),
        scopes: str = Form(None),
        profile: str = Form(None),
    ):
        require_admin(request)
        name = name.strip()[:MAX_NAME_LEN]
        if not name:
            raise HTTPException(400, "Token name is required")
        scope_list, profile_key = _service_profile_scopes(scopes, profile)
        try:
            owner = (
                validate_beepy_service_configuration()[0]
                if profile_key == BEEPY_SERVICE_TOKEN_PROFILE
                else get_current_user(request)
            )
        except KalConfigurationError as exc:
            raise HTTPException(503, str(exc)) from None
        scopes_value = ",".join(scope_list)

        raw_token = "ody_" + secrets.token_urlsafe(32)
        token_hash = bcrypt.hashpw(raw_token.encode(), bcrypt.gensalt()).decode()
        token_id = str(uuid.uuid4())[:8]

        with get_db_session() as db:
            db.add(ApiToken(
                id=token_id,
                owner=owner,
                name=name,
                token_hash=token_hash,
                token_prefix=raw_token[:8],
                scopes=scopes_value,
                profile=profile_key,
                is_active=True,
            ))
        _invalidate_cache(request)

        return {
            "id": token_id,
            "name": name,
            "owner": owner,
            "token": raw_token,
            "token_prefix": raw_token[:8],
            "scopes": scope_list,
            "profile": profile_key,
        }

    @router.patch("/tokens/{token_id}")
    async def update_token(request: Request, token_id: str):
        require_admin(request)
        current_user = get_current_user(request)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        with get_db_session() as db:
            token = db.query(ApiToken).filter(ApiToken.id == token_id).first()
            if not token:
                raise HTTPException(404, "Token not found")
            token_scopes = [
                s.strip() for s in (getattr(token, "scopes", "") or "").split(",") if s.strip()
            ]
            token_profile = getattr(token, "profile", None)
            service_marked = has_beepy_service_markers(
                getattr(token, "owner", None), token_scopes, token_profile
            )
            exact_service = is_exact_beepy_service_identity(
                getattr(token, "owner", None), token_scopes, token_profile
            )
            if service_marked and not exact_service:
                raise HTTPException(409, "Malformed Beepy service token profile")
            if current_user and token.owner != current_user and not exact_service:
                raise HTTPException(403, "Not your token")
            if isinstance(payload.get("name"), str) and payload["name"].strip():
                token.name = payload["name"].strip()[:MAX_NAME_LEN]
            # Only touch scopes when the caller actually sent them. A partial
            # update such as a rename ({"name": ...} with no "scopes" key) must
            # not silently reset the token to the default scope — that dropped
            # every previously granted scope.
            if exact_service and any(key in payload for key in ("owner", "profile", "scopes")):
                raise HTTPException(400, "Beepy service token identity is immutable")
            if "scopes" in payload:
                normalized, _ = _service_profile_scopes(payload.get("scopes"), None)
                token.scopes = ",".join(normalized)
            db.add(token)
            current_scopes = [
                s.strip()
                for s in (getattr(token, "scopes", "") or DEFAULT_SCOPES).split(",")
                if s.strip()
            ]
            response = {
                "id": token_id,
                "name": getattr(token, "name", ""),
                "owner": getattr(token, "owner", None),
                "token_prefix": getattr(token, "token_prefix", ""),
                "scopes": current_scopes,
                "profile": getattr(token, "profile", None),
            }
        _invalidate_cache(request)
        return response

    @router.delete("/tokens/{token_id}")
    def delete_token(request: Request, token_id: str):
        require_admin(request)
        current_user = get_current_user(request)
        with get_db_session() as db:
            token = db.query(ApiToken).filter(ApiToken.id == token_id).first()
            if not token:
                raise HTTPException(404, "Token not found")
            token_scopes = set((getattr(token, "scopes", "") or "").split(","))
            if current_user and token.owner != current_user and BEEPY_SHARED_RAG_SCOPE not in token_scopes:
                raise HTTPException(403, "Not your token")
            db.delete(token)
        _invalidate_cache(request)
        return {"status": "deleted"}

    return router
