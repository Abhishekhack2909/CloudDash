"""LangGraph Orchestrator — the core supervisor that manages agent flow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, StateGraph

from agents.billing_agent import BillingAgent
from agents.escalation_agent import EscalationAgent
from agents.technical_agent import TechnicalSupportAgent
from agents.triage_agent import TriageAgent
from guardrails.input_guardrails import InputGuardrails
from guardrails.output_guardrails import OutputGuardrails
from handover.protocol import HandoverProtocol
from models.conversation import ConversationState, Message, MessageRole
from observability.logger import get_logger
from retrieval.pipeline import RAGPipeline

logger = get_logger(__name__)


class Orchestrator:
    """LangGraph-based supervisor orchestrator for multi-agent routing.

    Implements the graph:
        input_guard → triage → (technical | billing | escalation) → output_guard → END

    Supports:
    - Conditional routing based on triage classification
    - Agent-to-agent handovers with context preservation
    - Guardrails on input and output
    - Max-turn protection against infinite loops
    - Handover audit logging
    """

    def __init__(self):
        # Initialize shared components
        self._rag = RAGPipeline()
        self._handover = HandoverProtocol()
        self._input_guard = InputGuardrails()
        self._output_guard = OutputGuardrails()

        # Initialize agents
        self._triage = TriageAgent()
        self._technical = TechnicalSupportAgent(rag_pipeline=self._rag)
        self._billing = BillingAgent(rag_pipeline=self._rag)
        self._escalation = EscalationAgent()

        # Build the graph
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        # Create graph with ConversationState as the state schema
        # We use dict-based state for LangGraph compatibility
        workflow = StateGraph(dict)

        # Add nodes
        workflow.add_node("input_guard", self._input_guard_node)
        workflow.add_node("triage", self._triage_node)
        workflow.add_node("technical_support", self._technical_node)
        workflow.add_node("billing", self._billing_node)
        workflow.add_node("escalation", self._escalation_node)
        workflow.add_node("output_guard", self._output_guard_node)

        # Set entry point
        workflow.set_entry_point("input_guard")

        # Add edges
        workflow.add_conditional_edges(
            "input_guard",
            self._route_after_input_guard,
            {
                "blocked": "output_guard",
                "triage": "triage",
            },
        )

        workflow.add_conditional_edges(
            "triage",
            self._route_after_triage,
            {
                "technical_support": "technical_support",
                "billing": "billing",
                "escalation": "escalation",
            },
        )

        workflow.add_conditional_edges(
            "technical_support",
            self._route_after_specialist,
            {
                "billing": "billing",
                "escalation": "escalation",
                "done": "output_guard",
            },
        )

        workflow.add_conditional_edges(
            "billing",
            self._route_after_specialist,
            {
                "technical_support": "technical_support",
                "escalation": "escalation",
                "done": "output_guard",
            },
        )

        workflow.add_edge("escalation", "output_guard")
        workflow.add_edge("output_guard", END)

        return workflow.compile()

    # --- Node functions (convert between dict and ConversationState) ---

    def _input_guard_node(self, state_dict: dict) -> dict:
        """Run input guardrails on the latest user message."""
        conv_state = ConversationState(**state_dict)
        user_message = ""
        for msg in reversed(conv_state.messages):
            if msg.role == MessageRole.USER:
                user_message = msg.content
                break

        result = self._input_guard.check(user_message)

        if not result.passed:
            conv_state.input_blocked = True
            conv_state.block_reason = result.reason
            conv_state.messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=result.sanitized_response,
                agent_id="guardrails",
            ))
            logger.warning(
                "input_blocked",
                conversation_id=conv_state.conversation_id,
                reason=result.reason,
            )

        return conv_state.model_dump()

    def _triage_node(self, state_dict: dict) -> dict:
        """Run the triage agent."""
        conv_state = ConversationState(**state_dict)
        updated = self._triage.process(conv_state)
        return updated.model_dump()

    def _technical_node(self, state_dict: dict) -> dict:
        """Run the technical support agent with handover tracking."""
        conv_state = ConversationState(**state_dict)

        # Log handover if coming from another agent
        if conv_state.previous_agent and conv_state.previous_agent != "triage":
            self._log_handover(conv_state, conv_state.previous_agent, "technical_support")

        conv_state.previous_agent = "technical_support"
        updated = self._technical.process(conv_state)
        return updated.model_dump()

    def _billing_node(self, state_dict: dict) -> dict:
        """Run the billing agent with handover tracking."""
        conv_state = ConversationState(**state_dict)

        if conv_state.previous_agent and conv_state.previous_agent != "triage":
            self._log_handover(conv_state, conv_state.previous_agent, "billing")

        conv_state.previous_agent = "billing"
        updated = self._billing.process(conv_state)
        return updated.model_dump()

    def _escalation_node(self, state_dict: dict) -> dict:
        """Run the escalation agent."""
        conv_state = ConversationState(**state_dict)

        if conv_state.previous_agent:
            self._log_handover(conv_state, conv_state.previous_agent, "escalation")

        updated = self._escalation.process(conv_state)
        return updated.model_dump()

    def _output_guard_node(self, state_dict: dict) -> dict:
        """Run output guardrails on the latest assistant response."""
        conv_state = ConversationState(**state_dict)

        # Find the last assistant message
        for msg in reversed(conv_state.messages):
            if msg.role == MessageRole.ASSISTANT:
                result = self._output_guard.check(
                    response=msg.content,
                    retrieved_contexts=[ctx.get("chunk_text", "") for ctx in conv_state.retrieved_contexts],
                )
                if result.modified:
                    msg.content = result.sanitized_response
                    msg.metadata["guardrail_applied"] = True
                    msg.metadata["guardrail_reasons"] = result.reasons
                    logger.info(
                        "output_guardrail_applied",
                        conversation_id=conv_state.conversation_id,
                        reasons=result.reasons,
                    )
                break

        return conv_state.model_dump()

    # --- Routing functions ---

    def _route_after_input_guard(self, state_dict: dict) -> str:
        """Route after input guardrails."""
        if state_dict.get("input_blocked"):
            return "blocked"
        return "triage"

    def _route_after_triage(self, state_dict: dict) -> str:
        """Route based on triage classification."""
        next_agent = state_dict.get("next_agent", "technical_support")
        valid_routes = {"technical_support", "billing", "escalation"}
        if next_agent in valid_routes:
            return next_agent
        return "technical_support"

    def _route_after_specialist(self, state_dict: dict) -> str:
        """Route after a specialist agent (technical/billing)."""
        next_agent = state_dict.get("next_agent")
        turn_count = state_dict.get("turn_count", 0)
        max_turns = state_dict.get("max_turns", 10)

        # Max turn protection
        if turn_count >= max_turns:
            logger.warning(
                "max_turns_reached",
                conversation_id=state_dict.get("conversation_id"),
                turn_count=turn_count,
            )
            return "escalation"

        # Route to next agent or finish
        if next_agent == "billing":
            return "billing"
        elif next_agent == "technical_support":
            return "technical_support"
        elif next_agent == "escalation":
            return "escalation"
        else:
            return "done"

    # --- Helper methods ---

    def _log_handover(self, state: ConversationState, source: str, target: str):
        """Log a handover event."""
        handover_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_agent": source,
            "target_agent": target,
            "reason": state.routing_reason or "Agent routing decision",
            "context_snapshot": {
                "message_count": len(state.messages),
                "entities": state.extracted_entities.model_dump(),
                "turn_count": state.turn_count,
            },
        }
        state.handover_history.append(handover_event)

        # Also log via the handover protocol
        self._handover.log_handover(
            conversation_id=state.conversation_id,
            source_agent=source,
            target_agent=target,
            reason=state.routing_reason or "Agent routing decision",
            context_snapshot=handover_event["context_snapshot"],
        )

        logger.info(
            "agent_handover",
            conversation_id=state.conversation_id,
            source_agent=source,
            target_agent=target,
            reason=state.routing_reason,
        )

    def process_message(
        self,
        conversation_id: str | None = None,
        message: str = "",
        existing_state: dict | None = None,
    ) -> dict:
        """Process a single user message through the agent pipeline.

        Args:
            conversation_id: Optional conversation ID (generated if not provided).
            message: The user's message.
            existing_state: Optional existing conversation state for multi-turn.

        Returns:
            Updated conversation state as a dict.
        """
        if existing_state:
            state = ConversationState(**existing_state)
        else:
            state = ConversationState()
            if conversation_id:
                state.conversation_id = conversation_id

        # Add the user message
        state.messages.append(Message(
            role=MessageRole.USER,
            content=message,
        ))

        logger.info(
            "processing_message",
            conversation_id=state.conversation_id,
            message_preview=message[:100],
            turn_count=state.turn_count,
        )

        # Reset routing state for new message
        state.next_agent = None
        state.routing_reason = None
        state.input_blocked = False
        state.block_reason = None
        state.is_resolved = False
        state.retrieved_contexts = []

        # Run through the graph
        result = self._graph.invoke(state.model_dump())

        return result

    def get_handover_logs(self, conversation_id: str) -> list[dict]:
        """Get all handover audit logs for a conversation."""
        return self._handover.get_logs(conversation_id)
