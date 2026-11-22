"""API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# --- Request Schemas ---

class CustomerProfile(BaseModel):
    """Customer profile for personalization."""

    plan_tier: Optional[str] = Field("Pro", description="Customer plan tier: Free, Starter, Pro, Enterprise")
    active_integrations: list[str] = Field(default_factory=list, description="Active user integrations, e.g. ['aws', 'slack', 'datadog']")
    account_age_days: Optional[int] = Field(30, description="Account age in days")


class StartConversationRequest(BaseModel):
    """Request to start a new conversation."""

    message: str = Field(..., description="Initial customer message", min_length=1)
    user_id: Optional[str] = Field(None, description="Optional user ID for scoped long-term memory")
    customer_id: Optional[str] = Field(None, description="Optional customer identifier")
    customer_profile: Optional[CustomerProfile] = Field(None, description="Optional customer profile for personalized retrieval")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SendMessageRequest(BaseModel):
    """Request to send a message in an existing conversation."""

    message: str = Field(..., description="Customer message", min_length=1)
    user_id: Optional[str] = Field(None, description="Optional user ID for scoped long-term memory")
    customer_profile: Optional[CustomerProfile] = Field(None, description="Optional customer profile for personalized retrieval")


class FeedbackRequest(BaseModel):
    """Request to submit feedback on an agent response."""

    message_id: Optional[str] = Field(None, description="Optional message ID")
    rating: int = Field(..., description="+1 for positive, -1 for negative")
    comment: Optional[str] = Field(None, description="Optional constructive feedback comment")


class MetricsResponse(BaseModel):
    """Operational platform metrics response."""

    total_conversations: int
    resolved_conversations: int
    resolution_rate: float
    escalated_conversations: int
    escalation_rate: float
    average_turns_to_resolution: float
    total_feedbacks: int
    positive_feedback_ratio: float
    escalation_by_category: dict[str, int] = Field(default_factory=dict)



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
    model_in_use: str = ""
    fallback_used: bool = False
    available_models: list[str] = []


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str = ""
    conversation_id: Optional[str] = None
