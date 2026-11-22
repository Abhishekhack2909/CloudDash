"""Conversation and handover audit log repository backed by SQLAlchemy."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from database.connection import get_db_session, init_db
from database.models import ConversationRecord, HandoverLogRecord, MessageRecord


class ConversationStore:
    """Manages persistent storage for conversations, messages, and handover logs."""

    def __init__(self):
        init_db()

    def save_conversation(
        self,
        conversation_id: str,
        state_dict: dict[str, Any],
        user_id: Optional[str] = None,
    ) -> None:
        """Upsert a conversation record."""
        # Ensure user_id from state if not directly passed
        resolved_user_id = user_id or state_dict.get("user_id") or state_dict.get("extracted_entities", {}).get("customer_id")
        serialized_state = json.dumps(state_dict, default=str)

        with get_db_session() as session:
            record = session.query(ConversationRecord).filter_by(id=conversation_id).first()
            if record:
                record.current_agent = state_dict.get("current_agent", "triage")
                record.is_resolved = state_dict.get("is_resolved", False)
                record.is_escalated = state_dict.get("is_escalated", False)
                record.requires_human = state_dict.get("requires_human", False)
                record.turn_count = state_dict.get("turn_count", 0)
                record.state_json = serialized_state
                if resolved_user_id:
                    record.user_id = resolved_user_id
                record.updated_at = datetime.now(timezone.utc)
            else:
                record = ConversationRecord(
                    id=conversation_id,
                    user_id=resolved_user_id,
                    current_agent=state_dict.get("current_agent", "triage"),
                    is_resolved=state_dict.get("is_resolved", False),
                    is_escalated=state_dict.get("is_escalated", False),
                    requires_human=state_dict.get("requires_human", False),
                    turn_count=state_dict.get("turn_count", 0),
                    state_json=serialized_state,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                session.add(record)

    def get_conversation(self, conversation_id: str) -> Optional[dict[str, Any]]:
        """Retrieve conversation state by ID."""
        with get_db_session() as session:
            record = session.query(ConversationRecord).filter_by(id=conversation_id).first()
            if not record:
                return None
            try:
                state = json.loads(record.state_json)
                if record.user_id:
                    state["user_id"] = record.user_id
                return state
            except Exception:
                return None

    def list_conversations(self, user_id: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        """List recent conversations, optionally filtered by user_id."""
        with get_db_session() as session:
            query = session.query(ConversationRecord)
            if user_id:
                query = query.filter_by(user_id=user_id)
            records = query.order_by(ConversationRecord.updated_at.desc()).limit(limit).all()
            results = []
            for r in records:
                results.append({
                    "id": r.id,
                    "user_id": r.user_id,
                    "current_agent": r.current_agent,
                    "is_resolved": r.is_resolved,
                    "is_escalated": r.is_escalated,
                    "turn_count": r.turn_count,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                })
            return results

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        agent_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        message_id: Optional[str] = None,
    ) -> str:
        """Save a message record associated with a conversation."""
        with get_db_session() as session:
            msg = MessageRecord(
                id=message_id or str(uuid.uuid4()) if "uuid" in globals() else str(datetime.now().timestamp()),
                conversation_id=conversation_id,
                role=role,
                content=content,
                agent_id=agent_id,
                metadata_json=json.dumps(metadata or {}, default=str),
                created_at=datetime.now(timezone.utc),
            )
            session.add(msg)
            return msg.id

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        """Retrieve all messages for a conversation ordered chronologically."""
        with get_db_session() as session:
            records = (
                session.query(MessageRecord)
                .filter_by(conversation_id=conversation_id)
                .order_by(MessageRecord.created_at.asc())
                .all()
            )
            return [
                {
                    "id": r.id,
                    "conversation_id": r.conversation_id,
                    "role": r.role,
                    "content": r.content,
                    "agent_id": r.agent_id,
                    "metadata": json.loads(r.metadata_json) if r.metadata_json else {},
                    "timestamp": r.created_at.isoformat() if r.created_at else None,
                }
                for r in records
            ]

    def save_handover_log(
        self,
        conversation_id: str,
        source_agent: str,
        target_agent: str,
        reason: str,
        status: str,
        context_snapshot: Optional[dict[str, Any]] = None,
        log_id: Optional[str] = None,
    ) -> str:
        """Persist a handover audit event."""
        import uuid
        with get_db_session() as session:
            entry = HandoverLogRecord(
                id=log_id or str(uuid.uuid4()),
                conversation_id=conversation_id,
                source_agent=source_agent,
                target_agent=target_agent,
                reason=reason,
                status=status,
                context_snapshot_json=json.dumps(context_snapshot or {}, default=str),
                timestamp=datetime.now(timezone.utc),
            )
            session.add(entry)
            return entry.id

    def get_handover_logs(self, conversation_id: str) -> list[dict[str, Any]]:
        """Retrieve all handover audit events for a conversation."""
        with get_db_session() as session:
            records = (
                session.query(HandoverLogRecord)
                .filter_by(conversation_id=conversation_id)
                .order_by(HandoverLogRecord.timestamp.asc())
                .all()
            )
            return [
                {
                    "id": r.id,
                    "conversation_id": r.conversation_id,
                    "source_agent": r.source_agent,
                    "target_agent": r.target_agent,
                    "reason": r.reason,
                    "status": r.status,
                    "context_snapshot": json.loads(r.context_snapshot_json) if r.context_snapshot_json else {},
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                }
                for r in records
            ]
