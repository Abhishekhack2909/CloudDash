"""Conversation and agent response data models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of the message sender."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in a conversation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedEntities(BaseModel):
    """Entities extracted from the conversation."""

    customer_id: Optional[str] = None
    plan_type: Optional[str] = None
    issue_type: Optional[str] = None
    product_references: list[str] = Field(default_factory=list)
    urgency: Optional[str] = None  # low, medium, high, critical
    sentiment: Optional[str] = None  # positive, neutral, frustrated, angry
    custom: dict[str, Any] = Field(default_factory=dict)


class ConversationState(BaseModel):
    """Full state of a conversation, passed through the agent graph."""

    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = Field(default_factory=list)
    current_agent: str = "triage"
    previous_agent: Optional[str] = None
    extracted_entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    handover_history: list[dict[str, Any]] = Field(default_factory=list)
    turn_count: int = 0
    max_turns: int = 10
    is_resolved: bool = False
    is_escalated: bool = False
    requires_human: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Internal routing
    next_agent: Optional[str] = None
    routing_reason: Optional[str] = None

    # RAG context
    retrieved_contexts: list[dict[str, Any]] = Field(default_factory=list)

    # Guardrail flags
    input_blocked: bool = False
    block_reason: Optional[str] = None


class Citation(BaseModel):
    """A citation reference to a KB article."""

    article_id: str
    article_title: str
    category: str
    relevance_score: float = 0.0


class AgentResponse(BaseModel):
    """Structured response from an agent."""

    agent_id: str
    agent_name: str
    content: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    suggested_next_agent: Optional[str] = None
    requires_escalation: bool = False
    handover_reason: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
