"""Structured JSON logging using structlog.

Every log entry includes:
- trace_id (conversation_id) for correlating all logs in a session
- agent_id for identifying which agent produced the log
- event_type for categorizing the log
- timestamp in ISO format
"""

import logging
import os
import sys

import structlog


def setup_logging(log_level: str | None = None):
    """Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR). Defaults to env var LOG_LEVEL.
    """
    level = log_level or os.getenv("LOG_LEVEL", "INFO")

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging for third-party libs
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger for the given module.

    Args:
        name: Module name (typically __name__).

    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(name)


# Auto-setup on import
setup_logging()
