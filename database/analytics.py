"""Feedback storage and platform operational metrics for CloudDash."""

from typing import Any, Optional
import json

from database.connection import get_db_session, init_db
from database.models import ConversationRecord, FeedbackRecord


def save_feedback(
    conversation_id: str,
    rating: int,
    message_id: Optional[str] = None,
    comment: Optional[str] = None,
) -> dict[str, Any]:
    """Store customer thumbs up (+1) or thumbs down (-1) feedback with optional comment."""
    init_db()
    norm_rating = 1 if rating > 0 else -1
    with get_db_session() as session:
        record = FeedbackRecord(
            conversation_id=conversation_id,
            message_id=message_id,
            rating=norm_rating,
            comment=comment,
        )
        session.add(record)
        session.commit()
        return {
            "id": record.id,
            "conversation_id": record.conversation_id,
            "rating": record.rating,
            "comment": record.comment,
            "created_at": record.created_at.isoformat(),
        }


def get_metrics() -> dict[str, Any]:
    """Compute aggregate platform operational metrics from database."""
    init_db()
    with get_db_session() as session:
        convs = session.query(ConversationRecord).all()
        feedbacks = session.query(FeedbackRecord).all()

        total_conversations = len(convs)
        if total_conversations == 0:
            return {
                "total_conversations": 0,
                "resolved_conversations": 0,
                "resolution_rate": 0.0,
                "escalated_conversations": 0,
                "escalation_rate": 0.0,
                "average_turns_to_resolution": 0.0,
                "positive_feedback_ratio": 1.0,
                "total_feedbacks": len(feedbacks),
                "escalation_by_category": {},
            }

        resolved_count = 0
        escalated_count = 0
        total_turns = 0
        escalation_by_category: dict[str, int] = {}

        for c in convs:
            total_turns += (c.turn_count or 0)
            if c.is_escalated:
                escalated_count += 1
                try:
                    state = json.loads(c.state_json or "{}")
                    issue_cat = state.get("extracted_entities", {}).get("issue_type") or c.current_agent or "general"
                except Exception:
                    issue_cat = c.current_agent or "general"
                escalation_by_category[issue_cat] = escalation_by_category.get(issue_cat, 0) + 1
            elif c.is_resolved:
                resolved_count += 1

        resolution_rate = round(resolved_count / total_conversations, 4) if total_conversations else 0.0
        escalation_rate = round(escalated_count / total_conversations, 4) if total_conversations else 0.0
        avg_turns = round(total_turns / total_conversations, 2) if total_conversations else 0.0

        pos_feedbacks = sum(1 for f in feedbacks if f.rating > 0)
        pos_ratio = round(pos_feedbacks / len(feedbacks), 4) if feedbacks else 1.0

        return {
            "total_conversations": total_conversations,
            "resolved_conversations": resolved_count,
            "resolution_rate": resolution_rate,
            "escalated_conversations": escalated_count,
            "escalation_rate": escalation_rate,
            "average_turns_to_resolution": avg_turns,
            "total_feedbacks": len(feedbacks),
            "positive_feedback_count": pos_feedbacks,
            "positive_feedback_ratio": pos_ratio,
            "escalation_by_category": escalation_by_category,
        }
