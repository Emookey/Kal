"""Dedicated stateless Beepy integration capability."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.beepy_service import retrieve_beepy_shared_knowledge
from src.endpoint_resolver import resolve_endpoint
from src.llm_core import llm_call_async
from src.retrieval_policy import is_beepy_service_request


logger = logging.getLogger(__name__)

BEEPY_CONTRACT_VERSION = "beepy-kal.v1"
MAX_AUDIT_VALUE = 128
_AUDIT_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")


class BeepyTechnicalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=8000)

    @field_validator("message")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in value):
            raise ValueError("message contains unsupported control characters")
        return value


def _require_beepy_service(request: Request) -> None:
    if not is_beepy_service_request(request):
        raise HTTPException(403, "Exact Beepy service credential required")


def _audit_value(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > MAX_AUDIT_VALUE or not _AUDIT_VALUE_RE.fullmatch(normalized):
        raise HTTPException(400, f"Invalid {field_name} audit metadata")
    return normalized


def setup_beepy_integration_routes(rag_manager: Any) -> APIRouter:
    """Build the exclusive provider-neutral Beepy service router."""

    router = APIRouter(prefix="/api/integrations/beepy", tags=["beepy_integration"])

    @router.get("/capabilities")
    def capabilities(request: Request):
        _require_beepy_service(request)
        return {
            "contract_version": BEEPY_CONTRACT_VERSION,
            "capability": "technical_shared_knowledge",
            "stateless": True,
            "grounding_states": ["grounded", "no_match", "degraded"],
        }

    @router.post("/technical-chat")
    async def technical_chat(
        request: Request,
        payload: BeepyTechnicalRequest,
        beepy_actor: str | None = Header(default=None, alias="X-Kal-Beepy-Actor"),
        correlation_id: str | None = Header(default=None, alias="X-Kal-Request-ID"),
        business_operation: str | None = Header(
            default=None, alias="X-Kal-Business-Operation"
        ),
    ):
        _require_beepy_service(request)
        actor = _audit_value(beepy_actor, "actor")
        request_id = _audit_value(correlation_id, "request identifier") or str(uuid.uuid4())
        operation = _audit_value(business_operation, "business operation")

        grounding = retrieve_beepy_shared_knowledge(rag_manager, payload.message)
        endpoint_url, model, headers = resolve_endpoint(
            "default", owner=None, shared_only=True
        )
        if not endpoint_url or not model:
            raise HTTPException(503, "Kal reasoning service is not configured")

        try:
            response = await llm_call_async(
                endpoint_url,
                model,
                grounding.messages,
                headers=headers,
                temperature=0.2,
                max_tokens=2048,
                workload="foreground",
            )
        except Exception as exc:
            logger.warning(
                "Beepy reasoning unavailable request_id=%s error=%s",
                request_id,
                type(exc).__name__,
            )
            raise HTTPException(502, "Kal reasoning service is unavailable") from None

        logger.info(
            "Beepy technical request request_id=%s actor=%s operation=%s grounding=%s",
            request_id,
            actor or "-",
            operation or "-",
            grounding.status,
        )
        return {
            "contract_version": BEEPY_CONTRACT_VERSION,
            "response": str(response or ""),
            "shared_knowledge_used": grounding.used_shared_knowledge,
            "grounding_status": grounding.status,
            "sources": grounding.sources,
            "request_id": request_id,
        }

    return router
