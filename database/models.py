"""SQLAlchemy declarative models for CloudDash persistent storage."""

from datetime import datetime, timezone
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)

from database.connection import Base


def utc_now():
    return datetime.now(timezone.utc)


class ConversationRecord(Base):
    """Stores full conversation metadata and serialized state."""

    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), index=True, nullable=True)
    current_agent = Column(String(64), default="triage")
    is_resolved = Column(Boolean, default=False)
    is_escalated = Column(Boolean, default=False)
    requires_human = Column(Boolean, default=False)
    turn_count = Column(Integer, default=0)
    state_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)


class MessageRecord(Base):
    """Stores individual conversation messages."""

    __tablename__ = "messages"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(64), index=True, nullable=False)
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    agent_id = Column(String(64), nullable=True)
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime(timezone=True), default=utc_now)


class HandoverLogRecord(Base):
    """Audit log for agent-to-agent handover events."""

    __tablename__ = "handover_audit_logs"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(64), index=True, nullable=False)
    source_agent = Column(String(64), nullable=False)
    target_agent = Column(String(64), nullable=False)
    reason = Column(Text, nullable=True)
    status = Column(String(32), nullable=False)
    context_snapshot_json = Column(Text, default="{}")
    timestamp = Column(DateTime(timezone=True), default=utc_now)


class UserMemoryRecord(Base):
    """Long-term durable memory scoped per user."""

    __tablename__ = "user_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), index=True, nullable=False)
    key = Column(String(64), nullable=False)
    value = Column(Text, nullable=False)
    source_conversation_id = Column(String(64), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    __table_args__ = (
        UniqueConstraint("user_id", "key", name="uq_user_memory_user_key"),
    )


class CheckpointRecord(Base):
    """LangGraph checkpoints for state machine resumption."""

    __tablename__ = "checkpoints"

    thread_id = Column(String(64), primary_key=True)
    checkpoint_ns = Column(String(64), primary_key=True, default="")
    checkpoint_id = Column(String(128), primary_key=True)
    parent_checkpoint_id = Column(String(128), nullable=True)
    type_name = Column(String(64), nullable=False)
    checkpoint_blob = Column(LargeBinary, nullable=False)
    metadata_type = Column(String(64), nullable=False)
    metadata_blob = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)


class CheckpointWriteRecord(Base):
    """LangGraph task writes associated with checkpoints."""

    __tablename__ = "checkpoint_writes"

    thread_id = Column(String(64), primary_key=True)
    checkpoint_ns = Column(String(64), primary_key=True, default="")
    checkpoint_id = Column(String(128), primary_key=True)
    task_id = Column(String(128), primary_key=True)
    idx = Column(Integer, primary_key=True)
    channel = Column(String(128), nullable=False)
    type_name = Column(String(64), nullable=False)
    value_blob = Column(LargeBinary, nullable=False)


class SupportTicketRecord(Base):
    """Real customer support tickets created by agents."""

    __tablename__ = "support_tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(32), unique=True, index=True, nullable=False)
    title = Column(String(256), nullable=False)
    description = Column(Text, nullable=False)
    priority = Column(String(32), default="medium")
    category = Column(String(64), default="general")
    status = Column(String(32), default="open")
    customer_id = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)


class AccountRecord(Base):
    """Customer account data for real account lookups."""

    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(64), unique=True, index=True, nullable=False)
    company_name = Column(String(128), nullable=False)
    plan_tier = Column(String(32), nullable=False)
    monitored_hosts = Column(Integer, default=10)
    monthly_spend = Column(Float, default=199.0)
    renewal_date = Column(String(32), nullable=False)
    status = Column(String(32), default="active")


class FeedbackRecord(Base):
    """User rating and comments for response evaluation."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(64), index=True, nullable=False)
    message_id = Column(String(64), nullable=True)
    rating = Column(Integer, nullable=False)  # +1 (thumbs up) or -1 (thumbs down)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

