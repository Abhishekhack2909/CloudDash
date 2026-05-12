"""Handover protocol — manages context transfer and audit logging between agents."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from models.handover import HandoverLog, HandoverResult, HandoverStatus
from observability.logger import get_logger

logger = get_logger(__name__)


class HandoverProtocol:
    """Manages agent-to-agent handover with context preservation and audit logging.

    Features:
    - Context packaging (full history or summary)
    - Entity transfer without loss
    - Graceful failure handling with fallback chain
    - Full audit trail for every handover event
    """

    def __init__(self):
        # In-memory audit store (keyed by conversation_id)
        self._audit_logs: dict[str, list[HandoverLog]] = {}

    def execute_handover(
        self,
        conversation_id: str,
        source_agent: str,
        target_agent: str,
        reason: str,
        context_summary: str = "",
        extracted_entities: dict[str, Any] | None = None,
        available_agents: list[str] | None = None,
    ) -> HandoverResult:
        """Execute a handover from source to target agent.

        Args:
            conversation_id: The conversation's unique ID.
            source_agent: The agent initiating the handover.
            target_agent: The intended receiving agent.
            reason: Why the handover is happening.
            context_summary: Summary of conversation so far.
            extracted_entities: Entities to transfer.
            available_agents: List of available agent IDs for fallback.

        Returns:
            HandoverResult with success/failure status and actual target.
        """
        if available_agents is None:
            available = ["triage", "technical_support", "billing", "escalation"]
        else:
            available = available_agents

        # Validate target agent
        if target_agent in available:
            status = HandoverStatus.SUCCESS
            actual_target = target_agent
            error = None
        else:
            # Fallback chain
            logger.warning(
                "handover_target_unavailable",
                conversation_id=conversation_id,
                target_agent=target_agent,
            )
            fallback_chain = ["triage", "escalation"]
            actual_target = None
            for fallback in fallback_chain:
                if fallback in available:
                    actual_target = fallback
                    break

            if actual_target:
                status = HandoverStatus.FALLBACK
                error = f"Target '{target_agent}' unavailable, fell back to '{actual_target}'"
            else:
                status = HandoverStatus.FAILED
                actual_target = source_agent
                error = f"All targets unavailable, staying with '{source_agent}'"

        # Create audit log entry
        log_entry = HandoverLog(
            conversation_id=conversation_id,
            source_agent=source_agent,
            target_agent=target_agent,
            reason=reason,
            status=status,
            context_snapshot={
                "summary": context_summary,
                "entities": extracted_entities or {},
            },
            error_message=error,
            fallback_agent=actual_target if status == HandoverStatus.FALLBACK else None,
        )

        # Store the log
        if conversation_id not in self._audit_logs:
            self._audit_logs[conversation_id] = []
        self._audit_logs[conversation_id].append(log_entry)

        logger.info(
            "handover_executed",
            conversation_id=conversation_id,
            source_agent=source_agent,
            target_agent=target_agent,
            actual_target=actual_target,
            status=status.value,
        )

        return HandoverResult(
            success=status != HandoverStatus.FAILED,
            status=status,
            source_agent=source_agent,
            target_agent=target_agent,
            actual_target=actual_target,
            log_entry=log_entry,
            error=error,
        )

    def log_handover(
        self,
        conversation_id: str,
        source_agent: str,
        target_agent: str,
        reason: str,
        context_snapshot: dict[str, Any] | None = None,
    ):
        """Log a handover event without executing it (for tracking purposes)."""
        log_entry = HandoverLog(
            conversation_id=conversation_id,
            source_agent=source_agent,
            target_agent=target_agent,
            reason=reason,
            status=HandoverStatus.SUCCESS,
            context_snapshot=context_snapshot or {},
        )

        if conversation_id not in self._audit_logs:
            self._audit_logs[conversation_id] = []
        self._audit_logs[conversation_id].append(log_entry)

    def get_logs(self, conversation_id: str) -> list[dict]:
        """Get all handover audit logs for a conversation."""
        logs = self._audit_logs.get(conversation_id, [])
        return [log.model_dump() for log in logs]

    def get_all_logs(self) -> dict[str, list[dict]]:
        """Get all handover audit logs across all conversations."""
        return {
            conv_id: [log.model_dump() for log in logs]
            for conv_id, logs in self._audit_logs.items()
        }
