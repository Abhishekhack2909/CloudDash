"""Technical Support Agent — resolves technical issues using KB articles."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from models.conversation import Citation, ConversationState
from observability.logger import get_logger
from retrieval.pipeline import RAGPipeline

logger = get_logger(__name__)


class TechnicalSupportAgent(BaseAgent):
    """Resolves technical issues using KB retrieval, troubleshooting, and code generation."""

    def __init__(self, rag_pipeline: RAGPipeline | None = None):
        super().__init__(agent_id="technical_support")
        self._rag = rag_pipeline or RAGPipeline()

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

        results = self._rag.retrieve(
            query=user_message,
            conversation_context=conversation_history,
            top_k=5,
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

        # Build prompt with KB context
        augmented_prompt = f"""{self.system_prompt}

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
