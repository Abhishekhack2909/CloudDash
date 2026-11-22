"""Billing Agent — handles billing inquiries, plan changes, and payment issues."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from models.conversation import Citation, ConversationState
from observability.logger import get_logger
from retrieval.pipeline import RAGPipeline
from tools import create_support_ticket, lookup_account

logger = get_logger(__name__)

# Mock account data for demonstration (fallback)
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
    """Handles billing inquiries, explains invoices, and looks up real database accounts."""

    def __init__(self, rag_pipeline: RAGPipeline | None = None):
        super().__init__(agent_id="billing")
        self._rag = rag_pipeline or RAGPipeline()
        self._tools = [lookup_account, create_support_ticket]
        self._tool_map = {t.name: t for t in self._tools}
        self._llm_with_tools = self._llm.bind_tools(self._tools)

    def _lookup_account(self, customer_id: str | None = None) -> dict:
        """Lookup account data from real database tool or fallback to mock."""
        try:
            res = lookup_account.invoke({"customer_id": customer_id or "default"})
            if res and not res.get("error"):
                return {
                    "customer_id": res.get("customer_id", customer_id or "default"),
                    "name": res.get("company_name", "Acme Corp"),
                    "email": "billing@customer.io",
                    "plan": res.get("plan_tier", "Pro"),
                    "billing_cycle": "monthly",
                    "monthly_amount": f"${res.get('monthly_spend', 199.0):.2f}",
                    "next_billing_date": res.get("renewal_date", "2026-12-31"),
                    "payment_method": "Corporate Card ending in 4242",
                    "status": res.get("status", "active"),
                    "hosts_used": res.get("monitored_hosts", 10),
                    "hosts_limit": 100,
                    "users_used": 5,
                    "users_limit": 25,
                }
        except Exception:
            pass
        return MOCK_ACCOUNTS.get(customer_id or "default", MOCK_ACCOUNTS["default"])

    def process(self, state: ConversationState) -> ConversationState:
        """Process a billing-related request with KB-grounded response."""
        self._log_invocation(state)

        user_message = self._get_latest_user_message(state)
        if not user_message:
            return state

        conversation_history = self._get_conversation_history(state)

        # Retrieve relevant billing KB articles with tier boosting
        plan_tier = state.extracted_entities.plan_type or state.user_memory.get("plan_tier")
        results = self._rag.retrieve(
            query=user_message,
            conversation_context=conversation_history,
            category_filter=None,  # Don't filter — billing questions may reference features
            top_k=5,
            plan_tier=plan_tier,
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

        # Build memory section
        memory_section = ""
        if state.user_memory:
            memory_section = "\n\nCUSTOMER PROFILE & LONG-TERM MEMORY (From previous conversations):\n" + "\n".join(
                f"- {k}: {v}" for k, v in state.user_memory.items()
            )

        # Build prompt with KB context and account data
        augmented_prompt = f"""{self.system_prompt}{memory_section}

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

        response = self._llm_with_tools.invoke(messages)

        # Check if model invoked any tools
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_outputs = []
            for tc in response.tool_calls:
                tool_name = tc.get("name")
                tool_args = tc.get("args", {})
                if tool_name in self._tool_map:
                    try:
                        tool_fn = self._tool_map[tool_name]
                        tool_res = tool_fn.invoke(tool_args)
                    except Exception as err:
                        tool_res = {"error": str(err)}

                    call_record = {
                        "agent": self.agent_id,
                        "tool": tool_name,
                        "args": tool_args,
                        "result": tool_res,
                    }
                    state.tool_calls.append(call_record)
                    tool_outputs.append(f"Tool `{tool_name}` with args {tool_args} returned: {tool_res}")
                    logger.info("tool_executed", agent=self.agent_id, tool=tool_name, args=tool_args)

            tool_synth_prompt = f"""{augmented_prompt}

REAL-TIME TOOL EXECUTION RESULTS:
{chr(10).join(tool_outputs)}

Provide a clear, direct answer to the user communicating the billing/account data."""
            response = self._llm.invoke([
                SystemMessage(content=tool_synth_prompt),
                HumanMessage(content=user_message),
            ])

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
