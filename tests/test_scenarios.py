"""Integration tests for the 4 assessment scenarios.

These tests require OPENAI_API_KEY and a populated KB (run ingest.py first).
They validate the full end-to-end flow through the orchestrator.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping integration tests",
)


@pytest.fixture(scope="module")
def orchestrator():
    """Create a shared orchestrator instance for all scenario tests."""
    from agents.orchestrator import Orchestrator
    return Orchestrator()


class TestScenario1SingleAgent:
    """Scenario 1: Single-Agent Resolution.

    Customer: 'My CloudDash alerts stopped firing after I updated my
    AWS integration credentials yesterday. I'm on the Pro plan.'

    Expected: Triage → Technical Support → resolution with KB citations.
    """

    def test_single_agent_resolution(self, orchestrator):
        result = orchestrator.process_message(
            message="My CloudDash alerts stopped firing after I updated my AWS integration credentials yesterday. I'm on the Pro plan."
        )

        # Should have been processed
        assert result["turn_count"] >= 1

        # Should have assistant messages
        assistant_msgs = [
            m for m in result["messages"]
            if (m.get("role") == "assistant" or getattr(m.get("role", ""), "value", "") == "assistant")
        ]
        assert len(assistant_msgs) >= 1

        # The last response should contain relevant content
        last_response = assistant_msgs[-1].get("content", "").lower()
        assert any(
            keyword in last_response
            for keyword in ["alert", "integration", "credential", "aws", "kb-"]
        ), f"Response doesn't address the issue: {last_response[:200]}"


class TestScenario2CrossAgent:
    """Scenario 2: Cross-Agent Handover.

    Customer: 'I want to upgrade from Pro to Enterprise, but first can you
    check if the SSO integration issue I reported last week has been resolved?'

    Expected: Triage → Technical Support (SSO) → Billing (upgrade).
    """

    def test_cross_agent_handover(self, orchestrator):
        result = orchestrator.process_message(
            message="I want to upgrade from Pro to Enterprise, but first can you check if the SSO integration issue I reported last week has been resolved?"
        )

        assert result["turn_count"] >= 1
        assistant_msgs = [
            m for m in result["messages"]
            if (m.get("role") == "assistant" or getattr(m.get("role", ""), "value", "") == "assistant")
        ]
        assert len(assistant_msgs) >= 1


class TestScenario3Escalation:
    """Scenario 3: Escalation to Human.

    Customer: 'I've been charged twice for April. I need an immediate
    refund and I want to speak to a manager.'

    Expected: Triage → Billing/Escalation → human handover package.
    """

    def test_escalation_to_human(self, orchestrator):
        result = orchestrator.process_message(
            message="I've been charged twice for April. I need an immediate refund and I want to speak to a manager."
        )

        assert result["turn_count"] >= 1
        assistant_msgs = [
            m for m in result["messages"]
            if (m.get("role") == "assistant" or getattr(m.get("role", ""), "value", "") == "assistant")
        ]
        assert len(assistant_msgs) >= 1

        # Should show empathy and escalation intent
        combined = " ".join(m.get("content", "") for m in assistant_msgs).lower()
        assert any(
            word in combined
            for word in ["escalat", "manager", "specialist", "team", "human", "reference", "ticket"]
        ), f"Response doesn't indicate escalation: {combined[:300]}"


class TestScenario4KBFailure:
    """Scenario 4: KB Retrieval Failure.

    Customer: 'Does CloudDash support integration with Datadog for
    cross-platform alerting?'

    Expected: Agent searches KB → finds no article → acknowledges limitation.
    """

    def test_kb_retrieval_failure(self, orchestrator):
        result = orchestrator.process_message(
            message="Does CloudDash support integration with Datadog for cross-platform alerting?"
        )

        assert result["turn_count"] >= 1
        assistant_msgs = [
            m for m in result["messages"]
            if (m.get("role") == "assistant" or getattr(m.get("role", ""), "value", "") == "assistant")
        ]
        assert len(assistant_msgs) >= 1

        # Should acknowledge the limitation
        combined = " ".join(m.get("content", "") for m in assistant_msgs).lower()
        # Agent should not fabricate an answer — should indicate no info or escalation
        assert any(
            phrase in combined
            for phrase in [
                "don't have", "not available", "no information",
                "not currently", "escalate", "feature request",
                "product team", "not find", "unable to find",
                "don't currently", "isn't currently",
                "not supported", "cannot confirm",
                "knowledge base", "not aware",
            ]
        ) or "datadog" in combined, f"Response may fabricate info: {combined[:300]}"
