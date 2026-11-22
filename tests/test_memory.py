"""Tests for persistent storage, LangGraph checkpointing, and long-term user memory."""

import uuid
import pytest
from unittest.mock import MagicMock

from database.checkpointer import PersistentCheckpointer
from database.conversation_store import ConversationStore
from database.memory import UserMemoryService, DurableFacts
from models.conversation import ConversationState, Message, MessageRole
from agents.orchestrator import Orchestrator


@pytest.fixture
def clean_user_ids():
    """Generate unique user IDs and clean up afterwards."""
    u1 = f"user_test_{uuid.uuid4().hex[:8]}"
    u2 = f"user_test_{uuid.uuid4().hex[:8]}"
    yield u1, u2
    service = UserMemoryService()
    service.clear_user_memories(u1)
    service.clear_user_memories(u2)


def test_user_memory_persistence(clean_user_ids):
    """Test saving, retrieving, and updating user memory in database."""
    user_a, _ = clean_user_ids
    service = UserMemoryService()

    service.save_memory(user_a, "plan_tier", "Enterprise")
    service.save_memory(user_a, "integrations_used", "AWS CloudWatch, Okta SSO")

    memories = service.get_memories(user_a)
    assert memories["plan_tier"] == "Enterprise"
    assert "AWS CloudWatch" in memories["integrations_used"]

    # Test update (upsert)
    service.save_memory(user_a, "plan_tier", "Enterprise Dedicated")
    updated = service.get_memories(user_a)
    assert updated["plan_tier"] == "Enterprise Dedicated"


def test_user_memory_scoping_no_cross_user_leakage(clean_user_ids):
    """Ensure memories are strictly scoped per user with zero cross-user leakage."""
    user_a, user_b = clean_user_ids
    service = UserMemoryService()

    service.save_memory(user_a, "plan_tier", "Enterprise")
    service.save_memory(user_a, "cloud_region", "us-east-1")

    service.save_memory(user_b, "plan_tier", "Starter")
    service.save_memory(user_b, "cloud_region", "eu-west-1")

    mem_a = service.get_memories(user_a)
    mem_b = service.get_memories(user_b)

    assert mem_a["plan_tier"] == "Enterprise"
    assert mem_a["cloud_region"] == "us-east-1"
    assert "eu-west-1" not in mem_a.values()

    assert mem_b["plan_tier"] == "Starter"
    assert mem_b["cloud_region"] == "eu-west-1"
    assert "Enterprise" not in mem_b.values()

    # Formatted prompt verification
    prompt_a = service.format_memory_prompt(user_a)
    prompt_b = service.format_memory_prompt(user_b)

    assert "Enterprise" in prompt_a
    assert "Starter" not in prompt_a
    assert "Starter" in prompt_b
    assert "Enterprise" not in prompt_b


def test_conversation_and_handover_persistence_across_restarts():
    """Verify conversations, messages, and handover logs survive simulated restarts."""
    conv_id = f"test_conv_{uuid.uuid4().hex[:8]}"
    store = ConversationStore()

    # Session 1: write data
    state_dict = {
        "conversation_id": conv_id,
        "current_agent": "technical_support",
        "turn_count": 2,
        "is_resolved": True,
        "is_escalated": False,
        "messages": [
            {"role": "user", "content": "How do I configure alerts?"},
            {"role": "assistant", "content": "Navigate to Alerts > Alert Rules [KB-001]."},
        ],
    }
    store.save_conversation(conv_id, state_dict, user_id="test_persisted_user")
    store.save_message(conv_id, "user", "How do I configure alerts?")
    store.save_message(conv_id, "assistant", "Navigate to Alerts > Alert Rules [KB-001].")
    store.save_handover_log(
        conversation_id=conv_id,
        source_agent="triage",
        target_agent="technical_support",
        reason="Technical question about alert configuration",
        status="success",
        context_snapshot={"turn": 1},
    )

    # Simulate restart by creating fresh store instance
    restarted_store = ConversationStore()
    recovered_state = restarted_store.get_conversation(conv_id)
    assert recovered_state is not None
    assert recovered_state["current_agent"] == "technical_support"
    assert recovered_state["user_id"] == "test_persisted_user"
    assert recovered_state["is_resolved"] is True

    messages = restarted_store.get_messages(conv_id)
    assert len(messages) >= 2
    assert messages[0]["content"] == "How do I configure alerts?"

    logs = restarted_store.get_handover_logs(conv_id)
    assert len(logs) >= 1
    assert logs[0]["source_agent"] == "triage"
    assert logs[0]["target_agent"] == "technical_support"


def test_langgraph_checkpointer_persistence():
    """Verify LangGraph PersistentCheckpointer writes and loads checkpoints."""
    checkpointer = PersistentCheckpointer()
    thread_id = f"thread_{uuid.uuid4().hex[:8]}"

    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = {
        "v": 1,
        "id": "cp_001",
        "ts": "2026-04-15T12:00:00Z",
        "channel_values": {"count": 42},
        "channel_versions": {"count": 1},
        "versions_seen": {},
    }
    metadata = {"source": "input", "step": 1}

    # Store checkpoint
    saved_cfg = checkpointer.put(config, checkpoint, metadata, new_versions={})
    assert saved_cfg["configurable"]["checkpoint_id"] == "cp_001"

    # Retrieve checkpoint using fresh checkpointer instance
    reloaded_checkpointer = PersistentCheckpointer()
    tuple_data = reloaded_checkpointer.get_tuple(config)
    assert tuple_data is not None
    assert tuple_data.checkpoint["channel_values"]["count"] == 42
    assert tuple_data.metadata["step"] == 1


def test_fact_stated_in_session_1_used_in_session_2(clean_user_ids):
    """Test that a durable fact stored in session 1 is injected and used in session 2."""
    user_id, _ = clean_user_ids
    memory_service = UserMemoryService()

    # Session 1: user registers plan tier as Enterprise
    memory_service.save_memory(user_id, "plan_tier", "Enterprise")
    memory_service.save_memory(user_id, "integrations_used", "AWS CloudWatch")

    # Session 2: brand new conversation for same user
    orchestrator = Orchestrator(user_memory=memory_service)
    new_conv_id = f"conv_s2_{uuid.uuid4().hex[:8]}"

    # Mock triage agent LLM for predictable fast routing
    triage_agent = orchestrator._triage
    original_triage_llm = triage_agent._llm
    mock_triage_llm = MagicMock()
    mock_triage_resp = MagicMock()
    mock_triage_resp.content = '{"classification": {"primary_intent": "technical", "confidence": 0.95}, "entities": {}, "routing": {"next_agent": "technical_support"}}'
    mock_triage_llm.invoke.return_value = mock_triage_resp
    triage_agent._llm = mock_triage_llm

    # Mock technical agent LLM to assert that memory is present in augmented prompt
    technical_agent = orchestrator._technical
    original_llm = technical_agent._llm

    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = "Because you are on the Enterprise plan, your data retention is 365 days [KB-013: Data Retention Policy]."
    mock_llm.invoke.return_value = mock_resp
    technical_agent._llm = mock_llm

    try:
        # User asks a retention question without repeating their plan tier
        result = orchestrator.process_message(
            conversation_id=new_conv_id,
            message="What is our metric retention period?",
            user_id=user_id,
        )

        # Assert memory was loaded and injected into state
        assert result.get("user_id") == user_id
        assert result.get("user_memory", {}).get("plan_tier") == "Enterprise"
        assert result.get("extracted_entities", {}).get("plan_type") == "Enterprise"

        # Verify prompt received the memory context
        called_messages = mock_llm.invoke.call_args[0][0]
        prompt_content = called_messages[0].content
        assert "Enterprise" in prompt_content
        assert "CUSTOMER PROFILE & LONG-TERM MEMORY" in prompt_content
    finally:
        triage_agent._llm = original_triage_llm
        technical_agent._llm = original_llm


def test_fact_extraction_with_structured_schema(clean_user_ids):
    """Test durable fact extraction with structured Pydantic schema using mocked LLM."""
    user_id, _ = clean_user_ids
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = '''{
        "plan_tier": "Pro",
        "integrations_used": ["AWS CloudWatch", "Datadog"],
        "recurring_issues": ["Webhook 500 error"],
        "preferences": ["Wire transfer billing"],
        "custom_facts": {"preferred_region": "us-east-1"}
    }'''
    mock_llm.invoke.return_value = mock_resp

    service = UserMemoryService(llm=mock_llm)
    messages = [
        {"role": "user", "content": "I am on the Pro plan and using AWS CloudWatch and Datadog."},
        {"role": "assistant", "content": "Got it, I can help you with those integrations."},
    ]

    extracted = service.extract_and_save_facts(user_id=user_id, conversation_id="conv_123", messages=messages)

    assert extracted["plan_tier"] == "Pro"
    assert "AWS CloudWatch" in extracted["integrations_used"]
    assert "Datadog" in extracted["integrations_used"]
    assert extracted["recurring_issues"] == "Webhook 500 error"
    assert extracted["preferences"] == "Wire transfer billing"
    assert extracted["preferred_region"] == "us-east-1"

