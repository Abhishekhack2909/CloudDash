"""Base agent class providing common functionality for all agents."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from langchain_groq import ChatGroq

from models.conversation import AgentResponse, ConversationState, Message, MessageRole
from observability.logger import get_logger

logger = get_logger(__name__)


def load_agent_config() -> dict[str, Any]:
    """Load agent configuration from YAML."""
    config_path = Path(__file__).parent.parent / "config" / "agents.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BaseAgent(ABC):
    """Abstract base class for all CloudDash support agents.

    Provides:
    - YAML-based configuration loading
    - LLM client initialization
    - Structured logging
    - Common message formatting
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._config = load_agent_config()
        self._agent_config = self._config.get("agents", {}).get(agent_id, {})
        self._llm_config = self._config.get("llm", {})

        self.name = self._agent_config.get("name", agent_id)
        self.system_prompt = self._agent_config.get("system_prompt", "")
        self.capabilities = self._agent_config.get("capabilities", [])
        self.routes_to = self._agent_config.get("routes_to", [])

        self._llm = ChatGroq(
            model=self._llm_config.get("model", "llama-3.3-70b-versatile"),
            temperature=self._llm_config.get("temperature", 0.3),
            max_tokens=self._llm_config.get("max_tokens", 1024),
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

    @abstractmethod
    def process(self, state: ConversationState) -> ConversationState:
        """Process the current conversation state and return updated state.

        Args:
            state: Current conversation state.

        Returns:
            Updated conversation state with agent's response.
        """
        ...

    def _get_conversation_history(self, state: ConversationState) -> str:
        """Format conversation history as a string for LLM context."""
        history_parts = []
        for msg in state.messages[-10:]:  # Last 10 messages for context window
            role_label = "Customer" if msg.role == MessageRole.USER else f"Agent ({msg.agent_id or 'system'})"
            history_parts.append(f"{role_label}: {msg.content}")
        return "\n".join(history_parts)

    def _get_latest_user_message(self, state: ConversationState) -> str:
        """Get the most recent user message from the conversation."""
        for msg in reversed(state.messages):
            if msg.role == MessageRole.USER:
                return msg.content
        return ""

    def _add_assistant_message(
        self, state: ConversationState, content: str, metadata: dict | None = None
    ) -> ConversationState:
        """Add an assistant message to the conversation state."""
        message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            agent_id=self.agent_id,
            metadata=metadata or {},
        )
        state.messages.append(message)
        return state

    def _log_invocation(self, state: ConversationState):
        """Log agent invocation with structured data."""
        logger.info(
            "agent_invoked",
            agent_id=self.agent_id,
            agent_name=self.name,
            conversation_id=state.conversation_id,
            turn_count=state.turn_count,
            message_count=len(state.messages),
        )
