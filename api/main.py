"""FastAPI application — CloudDash Multi-Agent Customer Support API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agents.orchestrator import Orchestrator
from api.schemas import (
    ConversationHistoryResponse,
    ConversationResponse,
    ErrorResponse,
    HandoverLogResponse,
    HealthResponse,
    MessageResponse,
    SendMessageRequest,
    StartConversationRequest,
)
from observability.logger import get_logger, setup_logging
from observability.tracing import configure_tracing, get_tracing_config

# Initialize logging and tracing
setup_logging()
configure_tracing()
logger = get_logger(__name__)

# Initialize the orchestrator
orchestrator = Orchestrator()

# In-memory conversation store
conversations: dict[str, dict] = {}

# Create FastAPI app
app = FastAPI(
    title="CloudDash Customer Support API",
    description="Multi-Agent Customer Support System for CloudDash — a cloud infrastructure monitoring platform.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health and component status."""
    tracing = get_tracing_config()
    try:
        from retrieval.vector_store import VectorStore
        vs = VectorStore()
        kb_count = vs.count
    except Exception:
        kb_count = 0

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        agents=["triage", "technical_support", "billing", "escalation"],
        kb_articles_count=kb_count,
        tracing_enabled=tracing["enabled"],
    )


@app.post(
    "/conversations",
    response_model=ConversationResponse,
    tags=["Conversations"],
    summary="Start a new conversation",
)
async def start_conversation(request: StartConversationRequest):
    """Start a new customer support conversation.

    Sends the initial message through the triage pipeline for intent
    classification and routing to the appropriate specialist agent.
    """
    try:
        logger.info("api_new_conversation", message_preview=request.message[:100])

        result = orchestrator.process_message(message=request.message)
        conversation_id = result["conversation_id"]
        conversations[conversation_id] = result

        # Extract response messages (assistant messages only)
        response_messages = _extract_messages(result)
        citations = _extract_citations(result)

        return ConversationResponse(
            conversation_id=conversation_id,
            messages=response_messages,
            current_agent=result.get("current_agent", "triage"),
            is_resolved=result.get("is_resolved", False),
            is_escalated=result.get("is_escalated", False),
            requires_human=result.get("requires_human", False),
            handover_count=len(result.get("handover_history", [])),
            citations=citations,
        )

    except Exception as e:
        logger.error("api_error", error=str(e), endpoint="start_conversation")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationResponse,
    tags=["Conversations"],
    summary="Send a message in an existing conversation",
)
async def send_message(conversation_id: str, request: SendMessageRequest):
    """Send a follow-up message in an existing conversation.

    The conversation state is preserved, allowing multi-turn interactions
    with full context awareness.
    """
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found",
        )

    try:
        existing_state = conversations[conversation_id]

        logger.info(
            "api_send_message",
            conversation_id=conversation_id,
            message_preview=request.message[:100],
        )

        result = orchestrator.process_message(
            conversation_id=conversation_id,
            message=request.message,
            existing_state=existing_state,
        )

        conversations[conversation_id] = result
        response_messages = _extract_messages(result)
        citations = _extract_citations(result)

        return ConversationResponse(
            conversation_id=conversation_id,
            messages=response_messages,
            current_agent=result.get("current_agent", "unknown"),
            is_resolved=result.get("is_resolved", False),
            is_escalated=result.get("is_escalated", False),
            requires_human=result.get("requires_human", False),
            handover_count=len(result.get("handover_history", [])),
            citations=citations,
        )

    except Exception as e:
        logger.error(
            "api_error",
            error=str(e),
            conversation_id=conversation_id,
            endpoint="send_message",
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/conversations/{conversation_id}",
    response_model=ConversationHistoryResponse,
    tags=["Conversations"],
    summary="Get conversation history",
)
async def get_conversation(conversation_id: str):
    """Retrieve the full conversation history including all messages and metadata."""
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found",
        )

    state = conversations[conversation_id]
    messages = _extract_messages(state, include_all=True)

    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=messages,
        extracted_entities=state.get("extracted_entities", {}),
        handover_history=state.get("handover_history", []),
        is_resolved=state.get("is_resolved", False),
        is_escalated=state.get("is_escalated", False),
        turn_count=state.get("turn_count", 0),
    )


@app.get(
    "/conversations/{conversation_id}/handovers",
    response_model=list[dict],
    tags=["Conversations"],
    summary="Get handover audit log",
)
async def get_handover_logs(conversation_id: str):
    """Retrieve the handover audit log for a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found",
        )

    logs = orchestrator.get_handover_logs(conversation_id)
    return logs


# --- Helper functions ---

def _extract_messages(state: dict, include_all: bool = False) -> list[MessageResponse]:
    """Extract messages from conversation state."""
    messages = []
    for msg_data in state.get("messages", []):
        if isinstance(msg_data, dict):
            role = msg_data.get("role", "assistant")
            # Handle enum values
            if hasattr(role, "value"):
                role = role.value
            if not include_all and role == "system":
                continue
            messages.append(MessageResponse(
                role=role,
                content=msg_data.get("content", ""),
                agent_id=msg_data.get("agent_id"),
                timestamp=str(msg_data.get("timestamp", "")),
                metadata=msg_data.get("metadata", {}),
            ))
    return messages


def _extract_citations(state: dict) -> list[dict]:
    """Extract citations from the conversation state."""
    citations = []
    for msg_data in state.get("messages", []):
        if isinstance(msg_data, dict):
            meta = msg_data.get("metadata", {})
            msg_citations = meta.get("citations", [])
            citations.extend(msg_citations)
    return citations


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
