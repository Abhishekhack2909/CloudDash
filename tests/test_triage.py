"""Tests for the Triage Agent — intent classification and routing."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.conversation import ConversationState, Message, MessageRole


class TestTriageClassification:
    """Test triage agent intent classification and routing."""

    def _create_state_with_message(self, message: str) -> ConversationState:
        """Helper to create a state with a single user message."""
        state = ConversationState()
        state.messages.append(Message(role=MessageRole.USER, content=message))
        return state

    @patch("agents.triage_agent.BaseAgent.__init__", return_value=None)
    def test_technical_intent_classification(self, mock_init):
        """Test that technical issues are classified correctly."""
        from agents.triage_agent import TriageAgent

        agent = TriageAgent.__new__(TriageAgent)
        agent.agent_id = "triage"
        agent.name = "Triage Agent"
        agent.system_prompt = ""
        agent.capabilities = []
        agent.routes_to = []

        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "classification": {"primary_intent": "technical", "confidence": 0.95},
            "entities": {"urgency": "high", "sentiment": "frustrated", "product_references": ["alerts", "AWS"]},
            "routing": {"next_agent": "technical_support", "reason": "Technical issue with alerts"},
            "initial_response": "I understand you're having issues with alerts."
        })
        agent._llm = MagicMock()
        agent._llm.invoke.return_value = mock_response

        state = self._create_state_with_message(
            "My CloudDash alerts stopped firing after I updated my AWS integration credentials."
        )
        result = agent.process(state)

        assert result.next_agent == "technical_support"
        assert result.extracted_entities.issue_type == "technical"

    @patch("agents.triage_agent.BaseAgent.__init__", return_value=None)
    def test_billing_intent_classification(self, mock_init):
        """Test that billing issues are classified correctly."""
        from agents.triage_agent import TriageAgent

        agent = TriageAgent.__new__(TriageAgent)
        agent.agent_id = "triage"
        agent.name = "Triage Agent"
        agent.system_prompt = ""
        agent.capabilities = []
        agent.routes_to = []

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "classification": {"primary_intent": "billing", "confidence": 0.92},
            "entities": {"urgency": "high", "sentiment": "angry"},
            "routing": {"next_agent": "billing", "reason": "Billing dispute"},
            "initial_response": "I understand you have a billing concern."
        })
        agent._llm = MagicMock()
        agent._llm.invoke.return_value = mock_response

        state = self._create_state_with_message("I've been charged twice for April.")
        result = agent.process(state)

        assert result.next_agent == "billing"
        assert result.extracted_entities.issue_type == "billing"

    @patch("agents.triage_agent.BaseAgent.__init__", return_value=None)
    def test_escalation_intent(self, mock_init):
        """Test that escalation requests are routed correctly."""
        from agents.triage_agent import TriageAgent

        agent = TriageAgent.__new__(TriageAgent)
        agent.agent_id = "triage"
        agent.name = "Triage Agent"
        agent.system_prompt = ""
        agent.capabilities = []
        agent.routes_to = []

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "classification": {"primary_intent": "escalation", "confidence": 0.98},
            "entities": {"urgency": "critical", "sentiment": "angry"},
            "routing": {"next_agent": "escalation", "reason": "Customer demands manager"},
            "initial_response": "I understand you want to speak with someone."
        })
        agent._llm = MagicMock()
        agent._llm.invoke.return_value = mock_response

        state = self._create_state_with_message("I want to speak to a manager right now!")
        result = agent.process(state)

        assert result.next_agent == "escalation"

    @patch("agents.triage_agent.BaseAgent.__init__", return_value=None)
    def test_malformed_llm_response_fallback(self, mock_init):
        """Test that malformed LLM responses fall back gracefully."""
        from agents.triage_agent import TriageAgent

        agent = TriageAgent.__new__(TriageAgent)
        agent.agent_id = "triage"
        agent.name = "Triage Agent"
        agent.system_prompt = ""
        agent.capabilities = []
        agent.routes_to = []

        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON at all."
        agent._llm = MagicMock()
        agent._llm.invoke.return_value = mock_response

        state = self._create_state_with_message("Help me with something")
        result = agent.process(state)

        # Should fall back to technical_support
        assert result.next_agent == "technical_support"
        assert len(result.messages) >= 2  # User message + fallback response
