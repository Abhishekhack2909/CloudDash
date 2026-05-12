"""Tests for the handover protocol — context preservation and audit logging."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from handover.protocol import HandoverProtocol
from models.handover import HandoverStatus


class TestHandoverProtocol:
    """Test handover execution, fallback, and audit logging."""

    def setup_method(self):
        """Set up a fresh handover protocol instance."""
        self.protocol = HandoverProtocol()

    def test_successful_handover(self):
        """A handover to an available agent should succeed."""
        result = self.protocol.execute_handover(
            conversation_id="conv-123",
            source_agent="technical_support",
            target_agent="billing",
            reason="Customer asked about pricing",
            context_summary="Customer was troubleshooting alerts, now asking about plan upgrade.",
            extracted_entities={"plan_type": "Pro", "urgency": "medium"},
        )

        assert result.success is True
        assert result.status == HandoverStatus.SUCCESS
        assert result.actual_target == "billing"
        assert result.source_agent == "technical_support"
        assert result.target_agent == "billing"

    def test_handover_fallback(self):
        """A handover to an unavailable agent should fall back."""
        result = self.protocol.execute_handover(
            conversation_id="conv-456",
            source_agent="technical_support",
            target_agent="nonexistent_agent",
            reason="Testing fallback",
            available_agents=["triage", "escalation"],
        )

        assert result.success is True
        assert result.status == HandoverStatus.FALLBACK
        assert result.actual_target in ["triage", "escalation"]
        assert result.error is not None

    def test_handover_total_failure(self):
        """A handover with no available agents should fail gracefully."""
        result = self.protocol.execute_handover(
            conversation_id="conv-789",
            source_agent="technical_support",
            target_agent="billing",
            reason="Testing complete failure",
            available_agents=[],  # No agents available
        )

        assert result.success is False
        assert result.status == HandoverStatus.FAILED
        assert result.actual_target == "technical_support"  # Stays with source

    def test_audit_log_created(self):
        """Every handover should create an audit log entry."""
        self.protocol.execute_handover(
            conversation_id="conv-audit",
            source_agent="triage",
            target_agent="technical_support",
            reason="Technical issue detected",
        )

        logs = self.protocol.get_logs("conv-audit")
        assert len(logs) == 1
        assert logs[0]["source_agent"] == "triage"
        assert logs[0]["target_agent"] == "technical_support"
        assert logs[0]["reason"] == "Technical issue detected"
        assert logs[0]["conversation_id"] == "conv-audit"
        assert "timestamp" in logs[0]

    def test_multiple_handovers_logged(self):
        """Multiple handovers in a conversation should all be logged."""
        conv_id = "conv-multi"

        self.protocol.execute_handover(
            conversation_id=conv_id,
            source_agent="triage",
            target_agent="technical_support",
            reason="Technical issue",
        )
        self.protocol.execute_handover(
            conversation_id=conv_id,
            source_agent="technical_support",
            target_agent="billing",
            reason="Billing question emerged",
        )
        self.protocol.execute_handover(
            conversation_id=conv_id,
            source_agent="billing",
            target_agent="escalation",
            reason="Customer frustrated",
        )

        logs = self.protocol.get_logs(conv_id)
        assert len(logs) == 3
        assert logs[0]["source_agent"] == "triage"
        assert logs[1]["source_agent"] == "technical_support"
        assert logs[2]["source_agent"] == "billing"

    def test_context_preserved_in_log(self):
        """Handover logs should preserve context snapshot."""
        self.protocol.execute_handover(
            conversation_id="conv-context",
            source_agent="technical_support",
            target_agent="billing",
            reason="Plan upgrade needed",
            context_summary="Customer on Pro plan needs Enterprise features.",
            extracted_entities={"plan_type": "Pro", "customer_id": "CUST-999"},
        )

        logs = self.protocol.get_logs("conv-context")
        context = logs[0]["context_snapshot"]
        assert context["summary"] == "Customer on Pro plan needs Enterprise features."
        assert context["entities"]["plan_type"] == "Pro"

    def test_separate_conversations_isolated(self):
        """Logs from different conversations should be isolated."""
        self.protocol.execute_handover(
            conversation_id="conv-A",
            source_agent="triage",
            target_agent="billing",
            reason="Billing",
        )
        self.protocol.execute_handover(
            conversation_id="conv-B",
            source_agent="triage",
            target_agent="technical_support",
            reason="Technical",
        )

        logs_a = self.protocol.get_logs("conv-A")
        logs_b = self.protocol.get_logs("conv-B")
        assert len(logs_a) == 1
        assert len(logs_b) == 1
        assert logs_a[0]["target_agent"] == "billing"
        assert logs_b[0]["target_agent"] == "technical_support"
