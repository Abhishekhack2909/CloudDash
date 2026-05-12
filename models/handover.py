"""Handover protocol data models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class HandoverStatus(str, Enum):
    """Status of a handover operation."""

    SUCCESS = "success"
    FAILED = "failed"
    FALLBACK = "fallback"


class HandoverPayload(BaseModel):
    """Payload for handing over conversation between agents."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_agent: str
    target_agent: str
    reason: str
    context_summary: str
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    conversation_history_length: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HandoverLog(BaseModel):
    """Audit log entry for a handover event."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_agent: str
    target_agent: str
    reason: str
    status: HandoverStatus
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    fallback_agent: Optional[str] = None


class HandoverResult(BaseModel):
    """Result of a handover operation."""

    success: bool
    status: HandoverStatus
    source_agent: str
    target_agent: str
    actual_target: str  # May differ from target if fallback occurred
    log_entry: HandoverLog
    error: Optional[str] = None


class EscalationPackage(BaseModel):
    """Complete package for human escalation."""

    ticket_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    customer_id: Optional[str] = None
    priority: str = "P3"  # P1, P2, P3, P4
    sentiment: str = "neutral"
    category: str = "general"
    summary: str = ""
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    attempted_resolutions: list[str] = Field(default_factory=list)
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    recommended_action: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
