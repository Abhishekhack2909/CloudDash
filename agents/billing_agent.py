"""Billing Agent — handles billing inquiries, plan changes, and payment issues."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from models.conversation import Citation, ConversationState
from observability.logger import get_logger
from retrieval.pipeline import RAGPipeline

logger = get_logger(__name__)

# Mock account data for demonstration
MOCK_ACCOUNTS = {
    "default": {
        "customer_id": "CUST-12345",
        "name": "Acme Corp",
        "email": "admin@acmecorp.io",
        "plan": "Pro",
        "billing_cycle": "monthly",
        "monthly_amount": "$199.00",
        "next_billing_date": "2026-05-01",
        "payment_method": "Visa ending in 4242",
        "status": "active",
        "hosts_used": 47,
        "hosts_limit": 100,
        "users_used": 12,
        "users_limit": 25,
    }
}


class BillingAgent(BaseAgent):
    """Handles billing inquiries, explains invoices, processes plan changes (simulated)."""

    def __init__(self, rag_pipeline: RAGPipeline | None = None):
        super().__init__(agent_id="billing")
        self._rag = rag_pipeline or RAGPipeline()

    def _lookup_account(self, customer_id: str | None = None) -> dict:
        """Mock account lookup. Returns simulated account data."""
        return MOCK_ACCOUNTS.get(customer_id or "default", MOCK_ACCOUNTS["default"])

    def process(self, state: ConversationState) -> ConversationState:
        """Process a billing-related request with KB-grounded response."""
        self._log_invocation(state)

        user_message = self._get_latest_user_message(state)
        if not user_message:
            return state

        conversation_history = self._get_conversation_history(state)

        # Retrieve relevant billing KB articles
        results = self._rag.retrieve(
            query=user_message,
            conversation_context=conversation_history,
            category_filter=None,  # Don't filter — billing questions may reference features
            top_k=5,
        )

        kb_context = self._rag.format_context(results)

        # Look up account info
        account = self._lookup_account(state.extracted_entities.customer_id)

        logger.info(
            "billing_kb_retrieval",
            conversation_id=state.conversation_id,
            results_count=len(results),
        )

        # Store retrieved contexts
        state.retrieved_contexts = [
            {
                "article_id": r.article_id,
                "article_title": r.article_title,
                "category": r.category,
                "relevance_score": r.relevance_score,
                "chunk_text": r.chunk_text[:200],
            }
            for r in results
        ]

        # Build prompt with KB context and account data
        augmented_prompt = f"""{self.system_prompt}

KNOWLEDGE BASE CONTEXT:
{kb_context}

CUSTOMER ACCOUNT DATA:
- Customer ID: {account['customer_id']}
- Company: {account['name']}
- Current Plan: {account['plan']}
- Billing Cycle: {account['billing_cycle']}
- Monthly Amount: {account['monthly_amount']}
- Next Billing Date: {account['next_billing_date']}
- Payment Method: {account['payment_method']}
- Account Status: {account['status']}
- Hosts: {account['hosts_used']}/{account['hosts_limit']}
- Users: {account['users_used']}/{account['users_limit']}

CONVERSATION HISTORY:
{conversation_history}

CUSTOMER QUERY: {user_message}

EXTRACTED INFORMATION:
- Urgency: {state.extracted_entities.urgency or 'Medium'}
- Sentiment: {state.extracted_entities.sentiment or 'Neutral'}

Provide a helpful response grounded in the KB articles and account data.
Cite billing policies using [KB-XXX: Title] format."""

        messages = [
            SystemMessage(content=augmented_prompt),
            HumanMessage(content=user_message),
        ]

        response = self._llm.invoke(messages)
        response_text = response.content.strip()

        # Parse handover/escalation signals
        handover_match = re.search(r'\[HANDOVER:\s*(\w+)\s*\|\s*(.+?)\]', response_text)
        escalate_match = re.search(r'\[ESCALATE:\s*(.+?)\]', response_text)

        if handover_match:
            target = handover_match.group(1)
            reason = handover_match.group(2)
            state.next_agent = target
            state.routing_reason = reason
            response_text = response_text.replace(handover_match.group(0), "").strip()
        elif escalate_match:
            reason = escalate_match.group(1)
            state.next_agent = "escalation"
            state.routing_reason = reason
            state.requires_human = True
            response_text = response_text.replace(escalate_match.group(0), "").strip()
        else:
            state.next_agent = None
            state.is_resolved = True

        # Extract citations
        citations = []
        citation_pattern = re.findall(r'\[KB-(\d+):\s*(.+?)\]', response_text)
        for kb_num, title in citation_pattern:
            article_id = f"KB-{kb_num}"
            matching = [r for r in results if r.article_id == article_id]
            citations.append(Citation(
                article_id=article_id,
                article_title=title.strip(),
                category=matching[0].category if matching else "billing",
                relevance_score=matching[0].relevance_score if matching else 0.0,
            ))

        self._add_assistant_message(state, response_text, metadata={
            "agent": self.agent_id,
            "citations": [c.model_dump() for c in citations],
            "account_lookup": True,
        })

        state.current_agent = self.agent_id
        state.turn_count += 1

        logger.info(
            "billing_response_complete",
            conversation_id=state.conversation_id,
            citations_count=len(citations),
            next_agent=state.next_agent,
            is_resolved=state.is_resolved,
        )

        return state
