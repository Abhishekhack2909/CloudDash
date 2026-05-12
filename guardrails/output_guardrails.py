"""Output guardrails — PII redaction and hallucination checking."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from observability.logger import get_logger

logger = get_logger(__name__)


class OutputGuardrailResult(BaseModel):
    """Result of output guardrail checks."""

    modified: bool = False
    sanitized_response: str = ""
    reasons: list[str] = []
    pii_found: list[str] = []


class OutputGuardrails:
    """Output guardrails for PII redaction and hallucination prevention.

    Two checks:
    1. PII Redaction — regex-based detection and masking of sensitive data
    2. Hallucination Check — ensures claims about pricing/policies are grounded in KB
    """

    def __init__(self):
        config_path = Path(__file__).parent.parent / "config" / "guardrails.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        self._output_config = self._config.get("output_guardrails", {})
        self._pii_config = self._output_config.get("pii_redaction", {})
        self._hallucination_config = self._output_config.get("hallucination_check", {})
        self._policy_config = self._output_config.get("policy_violations", {})

    def check(
        self,
        response: str,
        retrieved_contexts: list[str] | None = None,
    ) -> OutputGuardrailResult:
        """Run all output guardrails on the agent's response.

        Args:
            response: The agent's response text.
            retrieved_contexts: KB chunks used to generate the response.

        Returns:
            OutputGuardrailResult with potentially modified response.
        """
        modified = False
        reasons = []
        pii_found = []
        sanitized = response

        # Check 1: PII redaction
        if self._pii_config.get("enabled", True):
            sanitized, pii_types = self._redact_pii(sanitized)
            if pii_types:
                modified = True
                reasons.append(f"PII redacted: {', '.join(pii_types)}")
                pii_found = pii_types
                logger.info(
                    "output_pii_redacted",
                    pii_types=pii_types,
                )

        # Check 2: Hallucination check
        if self._hallucination_config.get("enabled", True) and retrieved_contexts:
            hallucination_result = self._check_hallucination(sanitized, retrieved_contexts)
            if hallucination_result["flagged"]:
                # Append disclaimer rather than blocking
                disclaimer = self._hallucination_config.get(
                    "disclaimer",
                    "\n\n*Note: Some information in this response may not be verified against our knowledge base. Please confirm critical details with our support team.*",
                )
                sanitized = sanitized + "\n\n" + disclaimer
                modified = True
                reasons.append("Potential ungrounded claim detected — disclaimer appended")

        # Check 3: Policy violations
        if self._policy_config.get("enabled", True):
            violation = self._check_policy_violations(sanitized)
            if violation:
                modified = True
                reasons.append(f"Policy violation detected: {violation}")
                logger.warning("output_policy_violation", violation=violation)

        return OutputGuardrailResult(
            modified=modified,
            sanitized_response=sanitized,
            reasons=reasons,
            pii_found=pii_found,
        )

    def _redact_pii(self, text: str) -> tuple[str, list[str]]:
        """Detect and redact PII using regex patterns.

        Returns:
            Tuple of (redacted_text, list_of_pii_types_found).
        """
        patterns = self._pii_config.get("patterns", {})
        replacement = self._pii_config.get("replacement", "[REDACTED]")
        pii_types_found = []

        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                pii_types_found.append(pii_type)
                text = re.sub(pattern, replacement, text)

        return text, pii_types_found

    def _check_hallucination(
        self,
        response: str,
        retrieved_contexts: list[str],
    ) -> dict[str, Any]:
        """Check if the response contains claims about grounded topics not in KB.

        Uses keyword matching to detect mentions of pricing, policies, etc.
        that should be grounded in retrieved KB content.
        """
        grounded_topics = self._hallucination_config.get("grounded_topics", [])
        response_lower = response.lower()
        context_text = " ".join(retrieved_contexts).lower()

        for topic in grounded_topics:
            if topic.lower() in response_lower:
                # Topic is mentioned in response — check if it's also in KB context
                if topic.lower() not in context_text:
                    # Mentioned a grounded topic but no KB backing
                    return {"flagged": True, "topic": topic}

        return {"flagged": False}

    def _check_policy_violations(self, text: str) -> str | None:
        """Check for blocked claims in the response."""
        blocked = self._policy_config.get("blocked_claims", [])
        text_lower = text.lower()

        for claim in blocked:
            if claim.lower() in text_lower:
                return claim

        return None
