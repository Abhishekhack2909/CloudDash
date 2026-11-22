"""Tests for Phase 5: Real tools with structured outputs and schema validation."""

import pytest
from pydantic import ValidationError

from tools import (
    CheckIntegrationStatusInput,
    CreateSupportTicketInput,
    LookupAccountInput,
    check_integration_status,
    create_support_ticket,
    lookup_account,
)
from models.conversation import ConversationState, Message, MessageRole
from agents.technical_agent import TechnicalSupportAgent
from agents.billing_agent import BillingAgent


def test_tool_execution_works():
    """Verify tool execution succeeds and returns structured real data."""
    # 1. Create support ticket
    ticket_res = create_support_ticket.invoke({
        "title": "Database connection dropouts",
        "description": "High latency observed on US-East-1 replica.",
        "priority": "high",
        "category": "technical",
    })
    assert "ticket_id" in ticket_res
    assert ticket_res["ticket_id"].startswith("TICKET-")
    assert ticket_res["status"] == "open"

    # 2. Check integration status
    integ_res = check_integration_status.invoke({
        "integration_name": "AWS",
        "customer_id": "cust_101",
    })
    assert integ_res["health"] == "healthy"
    assert integ_res["rate_limit_remaining"] > 0

    # Check unhealthy integration condition
    unhealthy_res = check_integration_status.invoke({
        "integration_name": "AWS",
        "customer_id": "cust_102",
    })
    assert unhealthy_res["health"] == "unhealthy"
    assert "ERR_AUTH_EXPIRED" in unhealthy_res["error_details"]

    # 3. Lookup account
    account_res = lookup_account.invoke({"customer_id": "cust_101"})
    assert account_res["company_name"] == "Acme Corp"
    assert account_res["plan_tier"] == "Enterprise"
    assert account_res["monitored_hosts"] == 250


def test_invalid_args_rejected():
    """Verify that invalid arguments are rejected by Pydantic validation."""
    # Empty title or too short description must raise ValidationError
    with pytest.raises(ValidationError):
        CreateSupportTicketInput(title="ab", description="short")

    with pytest.raises(ValidationError):
        CheckIntegrationStatusInput(integration_name="")

    with pytest.raises(ValidationError):
        LookupAccountInput(customer_id="")


def test_agent_tool_binding_and_execution():
    """Verify that agents have bound tools and process tool call records into state."""
    tech_agent = TechnicalSupportAgent()
    assert hasattr(tech_agent, "_tools")
    assert len(tech_agent._tools) >= 2
    tool_names = [t.name for t in tech_agent._tools]
    assert "check_integration_status" in tool_names
    assert "create_support_ticket" in tool_names

    billing_agent = BillingAgent()
    assert hasattr(billing_agent, "_tools")
    billing_tool_names = [t.name for t in billing_agent._tools]
    assert "lookup_account" in billing_tool_names
    assert "create_support_ticket" in billing_tool_names

    # Test direct simulated tool execution into state
    state = ConversationState()
    call_record = {
        "agent": "technical_support",
        "tool": "check_integration_status",
        "args": {"integration_name": "AWS", "customer_id": "cust_101"},
        "result": check_integration_status.invoke({"integration_name": "AWS", "customer_id": "cust_101"}),
    }
    state.tool_calls.append(call_record)
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0]["result"]["health"] == "healthy"


def test_agent_invokes_tool_when_prompted():
    """Verify that TechnicalSupportAgent executes tool when user asks about integration health."""
    tech_agent = TechnicalSupportAgent()
    state = ConversationState()
    state.extracted_entities.customer_id = "cust_102"
    state.messages.append(Message(
        role=MessageRole.USER,
        content="Is my AWS integration healthy? Please check integration status.",
    ))

    # Mock tool response if LLM generates tool_call or execute directly
    from unittest.mock import MagicMock
    mock_tool_call = MagicMock()
    mock_tool_call.tool_calls = [{
        "name": "check_integration_status",
        "args": {"integration_name": "AWS", "customer_id": "cust_102"},
        "id": "call_mock_1",
    }]
    tech_agent._llm_with_tools = MagicMock()
    tech_agent._llm_with_tools.invoke.return_value = mock_tool_call

    mock_synth = MagicMock()
    mock_synth.content = "Your AWS integration is currently unhealthy due to expired credentials."
    tech_agent._llm = MagicMock()
    tech_agent._llm.invoke.return_value = mock_synth

    updated_state = tech_agent.process(state)

    # Tool calls must be recorded
    assert len(updated_state.tool_calls) >= 1
    assert updated_state.tool_calls[0]["tool"] == "check_integration_status"
    assert updated_state.tool_calls[0]["result"]["health"] == "unhealthy"

