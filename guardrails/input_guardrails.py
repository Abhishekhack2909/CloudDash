"""Input guardrails — prompt injection detection and off-topic filtering."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from observability.logger import get_logger

logger = get_logger(__name__)


class GuardrailResult(BaseModel):
    """Result of an input guardrail check."""

    passed: bool
    reason: str = ""
    sanitized_response: str = ""
    checks_performed: list[str] = []


class InputGuardrails:
    """Input guardrails for prompt injection detection and off-topic filtering.

    Two layers of defense:
    1. Pattern matching — fast, catches known injection patterns
    2. Off-topic detection — keyword-based check for CloudDash relevance
    """

    def __init__(self):
        config_path = Path(__file__).parent.parent / "config" / "guardrails.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        self._input_config = self._config.get("input_guardrails", {})
        self._injection_config = self._input_config.get("prompt_injection", {})
        self._offtopic_config = self._input_config.get("off_topic", {})

    def check(self, user_input: str) -> GuardrailResult:
        """Run all input guardrails on the user's message.

        Args:
            user_input: The raw user input.

        Returns:
            GuardrailResult indicating if the input passed all checks.
        """
        checks_performed = []

        # Check 1: Prompt injection detection
        if self._injection_config.get("enabled", True):
            injection_result = self._check_injection(user_input)
            checks_performed.append("prompt_injection")
            if not injection_result["passed"]:
                logger.warning(
                    "input_guardrail_triggered",
                    check="prompt_injection",
                    pattern=injection_result.get("matched_pattern", ""),
                    input_preview=user_input[:50],
                )
                return GuardrailResult(
                    passed=False,
                    reason=f"Prompt injection detected: {injection_result.get('matched_pattern', 'suspicious pattern')}",
                    sanitized_response=self._injection_config.get(
                        "response",
                        "I can only help with CloudDash-related support inquiries.",
                    ),
                    checks_performed=checks_performed,
                )

        # Check 2: Off-topic detection (lenient — only blocks clearly irrelevant)
        if self._offtopic_config.get("enabled", True):
            offtopic_result = self._check_offtopic(user_input)
            checks_performed.append("off_topic")
            if not offtopic_result["passed"]:
                logger.info(
                    "input_guardrail_triggered",
                    check="off_topic",
                    input_preview=user_input[:50],
                )
                return GuardrailResult(
                    passed=False,
                    reason="Message appears unrelated to CloudDash support",
                    sanitized_response=self._offtopic_config.get(
                        "response",
                        "I'm the CloudDash support assistant. How can I help with CloudDash?",
                    ),
                    checks_performed=checks_performed,
                )

        return GuardrailResult(
            passed=True,
            reason="All checks passed",
            checks_performed=checks_performed,
        )

    def _check_injection(self, text: str) -> dict[str, Any]:
        """Check for prompt injection patterns."""
        patterns = self._injection_config.get("patterns", [])
        text_lower = text.lower()

        for pattern in patterns:
            if pattern.lower() in text_lower:
                return {"passed": False, "matched_pattern": pattern}

        return {"passed": True}

    def _check_offtopic(self, text: str) -> dict[str, Any]:
        """Check if the message is related to CloudDash.

        Uses a lenient approach — only blocks if:
        1. The message is very short (< 5 words) AND has no relevant keywords
        2. OR the message is clearly about an unrelated topic

        Most ambiguous messages are allowed through to let the agents handle them.
        """
        text_lower = text.lower()
        allowed_topics = self._offtopic_config.get("allowed_topics", [])

        # Short-circuit: always pass longer messages (they likely contain context)
        if len(text.split()) > 8:
            return {"passed": True}

        # Check for any relevant keyword
        for topic in allowed_topics:
            if topic.lower() in text_lower:
                return {"passed": True}

        # Common support-related words that should always pass
        support_words = [
            "help", "issue", "problem", "error", "fix", "how", "what", "why",
            "can", "please", "need", "want", "not working", "broken", "question",
            "charge", "pay", "account", "upgrade", "plan", "setup", "config",
        ]
        for word in support_words:
            if word in text_lower:
                return {"passed": True}

        # If we reach here with a very short message and no matches, it's likely off-topic
        if len(text.split()) < 5:
            return {"passed": False}

        # Default: let it through
        return {"passed": True}
