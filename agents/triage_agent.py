"""Triage Agent — first point of contact for intent classification and routing."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from models.conversation import ConversationState, ExtractedEntities
from observability.logger import get_logger

logger = get_logger(__name__)


class TriageAgent(BaseAgent):
    """Classifies customer intent, extracts entities, and routes to specialist agents."""

    def __init__(self):
        super().__init__(agent_id="triage")

    def process(self, state: ConversationState) -> ConversationState:
        """Classify intent and route the conversation."""
        self._log_invocation(state)

        user_message = self._get_latest_user_message(state)
        if not user_message:
            return state

        conversation_history = self._get_conversation_history(state)

        # Build the prompt
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Conversation history:\n{conversation_history}\n\nLatest customer message: {user_message}"),
        ]

        # Get LLM classification
        response = self._llm.invoke(messages)
        raw_response = response.content.strip()

        # Parse the JSON response
        try:
            # Try to extract JSON from the response
            json_str = raw_response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            parsed = json.loads(json_str)
            classification = parsed.get("classification", {})
            entities = parsed.get("entities", {})
            routing = parsed.get("routing", {})
            initial_response = parsed.get("initial_response", "")

        except (json.JSONDecodeError, IndexError):
            logger.warning(
                "triage_parse_error",
                conversation_id=state.conversation_id,
                raw_response=raw_response[:200],
            )
            # Fallback: route to technical support
            classification = {"primary_intent": "technical", "confidence": 0.5}
            entities = {}
            routing = {"next_agent": "technical_support", "reason": "Failed to parse — defaulting to technical support"}
            initial_response = "I'd be happy to help you with that. Let me connect you with the right specialist."

        # Update conversation state with extracted entities
        state.extracted_entities = ExtractedEntities(
            customer_id=entities.get("customer_id"),
            plan_type=entities.get("plan_type"),
            issue_type=classification.get("primary_intent"),
            product_references=entities.get("product_references", []),
            urgency=entities.get("urgency", "medium"),
            sentiment=entities.get("sentiment", "neutral"),
        )

        # Set routing
        next_agent = routing.get("next_agent", "technical_support")
        # Normalize agent names
        agent_map = {
            "technical": "technical_support",
            "tech": "technical_support",
            "technical_support": "technical_support",
            "billing": "billing",
            "escalation": "escalation",
            "account": "technical_support",
            "general": "technical_support",
        }
        state.next_agent = agent_map.get(next_agent, "technical_support")
        state.routing_reason = routing.get("reason", "")
        state.current_agent = "triage"

        # Add the triage response to conversation
        if initial_response:
            self._add_assistant_message(state, initial_response, metadata={
                "agent": "triage",
                "classification": classification,
                "routing": routing,
            })

        # Check for escalation signals
        if classification.get("primary_intent") == "escalation":
            state.next_agent = "escalation"

        logger.info(
            "triage_complete",
            conversation_id=state.conversation_id,
            primary_intent=classification.get("primary_intent"),
            next_agent=state.next_agent,
            confidence=classification.get("confidence"),
            sentiment=entities.get("sentiment"),
        )

        state.turn_count += 1
        return state
