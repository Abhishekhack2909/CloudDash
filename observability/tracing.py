"""LangSmith tracing configuration.

LangSmith provides native tracing for LangGraph/LangChain.
Auto-instrumentation is enabled via environment variables:
  LANGCHAIN_TRACING_V2=true
  LANGSMITH_API_KEY=<your-key>
  LANGSMITH_PROJECT=clouddash-support

This module provides helper utilities for custom trace metadata.
"""

import os
from functools import wraps
from typing import Any, Callable

from observability.logger import get_logger

logger = get_logger(__name__)


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is configured and enabled."""
    return (
        os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
        and bool(os.getenv("LANGSMITH_API_KEY"))
    )


def get_tracing_config() -> dict[str, Any]:
    """Get the current tracing configuration for display/logging."""
    return {
        "enabled": is_tracing_enabled(),
        "project": os.getenv("LANGSMITH_PROJECT", "default"),
        "api_key_set": bool(os.getenv("LANGSMITH_API_KEY")),
    }


def configure_tracing():
    """Ensure LangSmith tracing environment is properly set up.

    LangSmith auto-instruments LangChain/LangGraph calls when the
    environment variables are set. This function validates the config
    and logs the tracing status.
    """
    if is_tracing_enabled():
        project = os.getenv("LANGSMITH_PROJECT", "default")
        logger.info(
            "langsmith_tracing_enabled",
            project=project,
        )
    else:
        logger.info(
            "langsmith_tracing_disabled",
            hint="Set LANGCHAIN_TRACING_V2=true and LANGSMITH_API_KEY to enable",
        )


def trace_metadata(
    conversation_id: str = "",
    agent_id: str = "",
    **kwargs,
) -> dict[str, Any]:
    """Create metadata dict for attaching to LangSmith traces.

    Pass this as `metadata` to LangChain/LangGraph calls:
        llm.invoke(prompt, metadata=trace_metadata(conversation_id=..., agent_id=...))
    """
    meta = {
        "conversation_id": conversation_id,
        "agent_id": agent_id,
    }
    meta.update(kwargs)
    return meta
