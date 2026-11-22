"""Technical Support Agent — resolves technical issues using KB articles."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from models.conversation import Citation, ConversationState
from observability.logger import get_logger
from retrieval.pipeline import RAGPipeline
from tools import check_integration_status, create_support_ticket

logger = get_logger(__name__)


class TechnicalSupportAgent(BaseAgent):
    """Resolves technical issues using KB retrieval, live tools, and troubleshooting."""

    def __init__(self, rag_pipeline: RAGPipeline | None = None):
        super().__init__(agent_id="technical_support")
        self._rag = rag_pipeline or RAGPipeline()
        self._tools = [check_integration_status, create_support_ticket]
        self._tool_map = {t.name: t for t in self._tools}
        self._llm_with_tools = self._llm.bind_tools(self._tools)

    def process(self, state: ConversationState) -> ConversationState:
        """Process a technical support request with RAG-grounded response."""
        self._log_invocation(state)

        user_message = self._get_latest_user_message(state)
        if not user_message:
            return state

        conversation_history = self._get_conversation_history(state)

        # Retrieve relevant KB articles
        logger.info(
            "kb_retrieval_start",
            conversation_id=state.conversation_id,
            agent_id=self.agent_id,
            query=user_message[:100],
        )

        # Extract plan tier and integrations from state/user memory for personalized retrieval
        plan_tier = state.extracted_entities.plan_type or state.user_memory.get("plan_tier")
        active_integrations = []
        if state.user_memory.get("integrations_used"):
            active_integrations.extend([x.strip() for x in state.user_memory["integrations_used"].split(",")])
        if state.extracted_entities.product_references:
            active_integrations.extend(state.extracted_entities.product_references)

        results = self._rag.retrieve(
            query=user_message,
            conversation_context=conversation_history,
            top_k=5,
            plan_tier=plan_tier,
            active_integrations=active_integrations,
        )

        kb_context = self._rag.format_context(results)

        logger.info(
            "kb_retrieval_complete",
            conversation_id=state.conversation_id,
            agent_id=self.agent_id,
            results_count=len(results),
            top_articles=[r.article_id for r in results[:3]],
        )

        # Store retrieved contexts in state
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

        # Build prompt with KB context
        augmented_prompt = f"""{self.system_prompt}{memory_section}

KNOWLEDGE BASE CONTEXT:
{kb_context}

CONVERSATION HISTORY:
{conversation_history}

CUSTOMER QUERY: {user_message}

EXTRACTED INFORMATION:
- Customer Plan: {state.extracted_entities.plan_type or 'Unknown'}
- Issue Type: {state.extracted_entities.issue_type or 'Technical'}
- Urgency: {state.extracted_entities.urgency or 'Medium'}

Provide a helpful, accurate response grounded in the KB articles above. 
Always cite sources using [KB-XXX: Title] format.
If the KB doesn't contain relevant information, acknowledge this transparently."""

        messages = [
            SystemMessage(content=augmented_prompt),
            HumanMessage(content=user_message),
        ]

        # Invoke model with tools
        response = self._llm_with_tools.invoke(messages)

        # Check if the LLM invoked any tools
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

            # Synthesize final user-facing response grounded in tool output
            tool_synth_prompt = f"""{augmented_prompt}

REAL-TIME TOOL EXECUTION RESULTS:
{chr(10).join(tool_outputs)}

Provide a clear, direct answer to the user communicating the live status and any recommended steps."""
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
            # Clean the signal from the response
            response_text = response_text.replace(handover_match.group(0), "").strip()
        elif escalate_match:
            reason = escalate_match.group(1)
            state.next_agent = "escalation"
            state.routing_reason = reason
            state.requires_human = True
            response_text = response_text.replace(escalate_match.group(0), "").strip()
        else:
            state.next_agent = None  # No further routing needed
            state.is_resolved = True

        # Extract citations from response
        citations = []
        citation_pattern = re.findall(r'\[KB-(\d+):\s*(.+?)\]', response_text)
        for kb_num, title in citation_pattern:
            article_id = f"KB-{kb_num}"
            matching_results = [r for r in results if r.article_id == article_id]
            citations.append(Citation(
                article_id=article_id,
                article_title=title.strip(),
                category=matching_results[0].category if matching_results else "unknown",
                relevance_score=matching_results[0].relevance_score if matching_results else 0.0,
            ))

        # Add response to conversation
        self._add_assistant_message(state, response_text, metadata={
            "agent": self.agent_id,
            "citations": [c.model_dump() for c in citations],
            "kb_articles_used": len(results),
        })

        state.current_agent = self.agent_id
        state.turn_count += 1

        logger.info(
            "technical_response_complete",
            conversation_id=state.conversation_id,
            citations_count=len(citations),
            next_agent=state.next_agent,
            is_resolved=state.is_resolved,
        )

        return state
