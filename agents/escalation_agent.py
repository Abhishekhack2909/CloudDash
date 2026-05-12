"""Escalation Agent — packages context for human handover."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from models.conversation import ConversationState, MessageRole
from models.handover import EscalationPackage
from observability.logger import get_logger

logger = get_logger(__name__)


class EscalationAgent(BaseAgent):
    """Packages conversation context and routes to human support."""

    def __init__(self):
        super().__init__(agent_id="escalation")

    def process(self, state: ConversationState) -> ConversationState:
        """Create an escalation package and customer-facing message."""
        self._log_invocation(state)

        conversation_history = self._get_conversation_history(state)
        user_message = self._get_latest_user_message(state)

        # Build prompt for escalation summary
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Conversation history:
{conversation_history}

Latest customer message: {user_message}

Customer entities:
- Customer ID: {state.extracted_entities.customer_id or 'Unknown'}
- Plan: {state.extracted_entities.plan_type or 'Unknown'}
- Issue Type: {state.extracted_entities.issue_type or 'Unknown'}
- Urgency: {state.extracted_entities.urgency or 'medium'}
- Sentiment: {state.extracted_entities.sentiment or 'neutral'}

Previous agents involved: {', '.join([h.get('source_agent', '') for h in state.handover_history]) or 'None'}
Routing reason for escalation: {state.routing_reason or 'Customer requested human support'}

Generate the escalation summary and customer-facing message."""),
        ]

        response = self._llm.invoke(messages)
        raw_response = response.content.strip()

        # Parse the escalation response
        try:
            json_str = raw_response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            parsed = json.loads(json_str)
            escalation_data = parsed.get("escalation_summary", {})
            customer_message = parsed.get("customer_message", "")
        except (json.JSONDecodeError, IndexError):
            logger.warning(
                "escalation_parse_error",
                conversation_id=state.conversation_id,
                raw_response=raw_response[:200],
            )
            escalation_data = {
                "issue_summary": "Customer issue requires human intervention.",
                "attempted_resolutions": [],
                "escalation_reason": state.routing_reason or "Unable to resolve via AI agents",
                "priority": "P2",
                "sentiment": state.extracted_entities.sentiment or "neutral",
                "recommended_action": "Review conversation history and contact customer.",
            }
            customer_message = (
                "I understand your concern, and I want to make sure you get the best possible help. "
                "I'm escalating your case to a human support specialist who will be able to assist you further. "
                "You should hear from them within 2-4 hours during business hours. "
                f"Your reference number is ESC-{state.conversation_id[:8].upper()}."
            )

        # Build escalation package
        attempted_resolutions = []
        for msg in state.messages:
            if msg.role == MessageRole.ASSISTANT and msg.agent_id != "triage":
                attempted_resolutions.append(f"[{msg.agent_id}]: {msg.content[:100]}...")

        escalation_package = EscalationPackage(
            conversation_id=state.conversation_id,
            customer_id=state.extracted_entities.customer_id,
            priority=escalation_data.get("priority", "P3"),
            sentiment=escalation_data.get("sentiment", "neutral"),
            category=state.extracted_entities.issue_type or "general",
            summary=escalation_data.get("issue_summary", ""),
            conversation_history=[
                {"role": msg.role.value, "content": msg.content, "agent": msg.agent_id}
                for msg in state.messages
            ],
            attempted_resolutions=escalation_data.get("attempted_resolutions", attempted_resolutions),
            extracted_entities=state.extracted_entities.model_dump(),
            recommended_action=escalation_data.get("recommended_action", ""),
        )

        # Add customer-facing message
        if not customer_message:
            customer_message = (
                f"I've escalated your case to our support team. "
                f"Your reference number is ESC-{state.conversation_id[:8].upper()}. "
                f"A specialist will reach out to you shortly."
            )

        self._add_assistant_message(state, customer_message, metadata={
            "agent": self.agent_id,
            "escalation_package": escalation_package.model_dump(),
            "ticket_id": str(escalation_package.ticket_id),
        })

        # Mark conversation as escalated
        state.is_escalated = True
        state.requires_human = True
        state.is_resolved = False
        state.next_agent = None  # Terminal state
        state.current_agent = self.agent_id
        state.turn_count += 1

        logger.info(
            "escalation_complete",
            conversation_id=state.conversation_id,
            ticket_id=str(escalation_package.ticket_id),
            priority=escalation_package.priority,
            sentiment=escalation_package.sentiment,
            summary=escalation_package.summary[:100],
        )

        return state
