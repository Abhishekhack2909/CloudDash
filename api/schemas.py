"""API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# --- Request Schemas ---

class StartConversationRequest(BaseModel):
    """Request to start a new conversation."""

    message: str = Field(..., description="Initial customer message", min_length=1)
    customer_id: Optional[str] = Field(None, description="Optional customer identifier")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SendMessageRequest(BaseModel):
    """Request to send a message in an existing conversation."""

    message: str = Field(..., description="Customer message", min_length=1)


# --- Response Schemas ---

class MessageResponse(BaseModel):
    """A single message in the response."""

    role: str
    content: str
    agent_id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationResponse(BaseModel):
    """Response after processing a message."""

    conversation_id: str
    messages: list[MessageResponse]
    current_agent: str
    is_resolved: bool = False
    is_escalated: bool = False
    requires_human: bool = False
    handover_count: int = 0
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationHistoryResponse(BaseModel):
    """Full conversation history."""

    conversation_id: str
    messages: list[MessageResponse]
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    handover_history: list[dict[str, Any]] = Field(default_factory=list)
    is_resolved: bool = False
    is_escalated: bool = False
    turn_count: int = 0


class HandoverLogResponse(BaseModel):
    """Handover audit log entry."""

    id: str
    conversation_id: str
    timestamp: str
    source_agent: str
    target_agent: str
    reason: str
    status: str
    error_message: Optional[str] = None
    fallback_agent: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    agents: list[str] = []
    kb_articles_count: int = 0
    tracing_enabled: bool = False


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str = ""
    conversation_id: Optional[str] = None
