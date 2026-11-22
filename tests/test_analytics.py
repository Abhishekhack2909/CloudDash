"""Tests for Phase 7: Feedback collection and platform analytics metrics."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from database.analytics import get_metrics, save_feedback
from database.connection import get_db_session, init_db
from database.models import ConversationRecord, FeedbackRecord


@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    yield


def test_feedback_storage():
    """Verify thumbs up (+1) and thumbs down (-1) feedback storage."""
    conv_id = "test_conv_fb_1"
    
    # Positive feedback
    fb1 = save_feedback(
        conversation_id=conv_id,
        message_id="msg_1",
        rating=1,
        comment="Very helpful answer, resolved my issue quickly!",
    )
    assert fb1["conversation_id"] == conv_id
    assert fb1["rating"] == 1
    assert fb1["comment"] == "Very helpful answer, resolved my issue quickly!"

    # Negative feedback
    fb2 = save_feedback(
        conversation_id=conv_id,
        message_id="msg_2",
        rating=-1,
        comment="Did not explain step 3 clearly.",
    )
    assert fb2["rating"] == -1


def test_metrics_calculation():
    """Verify aggregate operational metrics calculation."""
    import uuid
    id1 = f"test_m_conv_{uuid.uuid4().hex[:8]}"
    id2 = f"test_m_conv_{uuid.uuid4().hex[:8]}"

    # Seed a resolved and an escalated conversation
    with get_db_session() as session:
        session.add(ConversationRecord(
            id=id1,
            current_agent="technical_support",
            is_resolved=True,
            is_escalated=False,
            turn_count=2,
            state_json='{"extracted_entities": {"issue_type": "technical"}}',
        ))
        session.add(ConversationRecord(
            id=id2,
            current_agent="escalation",
            is_resolved=False,
            is_escalated=True,
            turn_count=3,
            state_json='{"extracted_entities": {"issue_type": "billing_dispute"}}',
        ))
        session.commit()

    metrics = get_metrics()
    assert metrics["total_conversations"] >= 2
    assert metrics["resolution_rate"] >= 0.0
    assert metrics["escalation_rate"] >= 0.0
    assert metrics["average_turns_to_resolution"] >= 0.0
    assert "positive_feedback_ratio" in metrics


def test_api_metrics_endpoint():
    """Verify GET /metrics returns HTTP 200 with MetricsResponse schema."""
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_conversations" in data
    assert "resolution_rate" in data
    assert "escalation_rate" in data
    assert "positive_feedback_ratio" in data


def test_api_feedback_endpoint():
    """Verify POST /conversations/{id}/feedback saves feedback and returns 200."""
    client = TestClient(app)
    payload = {
        "message_id": "msg_test_api",
        "rating": 1,
        "comment": "Super clear instructions!",
    }
    response = client.post("/conversations/conv_test_api_123/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["feedback"]["rating"] == 1
