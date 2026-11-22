"""Long-term per-user memory module for extracting and injecting customer profile facts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config.llm import (
    DEFAULT_GROQ_MODEL,
    FallbackChatGroq,
    get_fallback_models,
    groq_model_options,
    load_agent_config,
)
from database.connection import get_db_session, init_db
from database.models import UserMemoryRecord
from observability.logger import get_logger

logger = get_logger(__name__)


class DurableFacts(BaseModel):
    """Structured schema for durable customer facts extracted from conversation."""

    plan_tier: Optional[str] = Field(
        None,
        description="Customer subscription tier: Starter, Pro, or Enterprise (or None if unmentioned).",
    )
    integrations_used: list[str] = Field(
        default_factory=list,
        description="Cloud or third-party integrations explicitly used by the customer (e.g. AWS CloudWatch, Okta, Slack, GCP).",
    )
    recurring_issues: list[str] = Field(
        default_factory=list,
        description="Persistent or recurring technical/operational pain points mentioned.",
    )
    preferences: list[str] = Field(
        default_factory=list,
        description="Specific stated customer preferences (e.g. wire transfer billing, Python SDK, us-east-1 region).",
    )
    custom_facts: dict[str, str] = Field(
        default_factory=dict,
        description="Any other durable facts worth remembering about this customer across sessions.",
    )


class UserMemoryService:
    """Manages long-term user memories and profile extraction."""

    def __init__(self, llm: Optional[FallbackChatGroq] = None):
        init_db()
        self._llm = llm

    def _get_llm(self) -> FallbackChatGroq:
        """Lazy initialization of LLM client for extraction."""
        if self._llm is None:
            from langchain_groq import ChatGroq
            import os

            configured_model = load_agent_config().get("llm", {}).get("model", DEFAULT_GROQ_MODEL)
            primary_opts = groq_model_options(configured_model)
            primary_client = ChatGroq(
                **primary_opts,
                temperature=0.0,
                groq_api_key=os.getenv("GROQ_API_KEY"),
            )
            self._llm = FallbackChatGroq(
                primary_client=primary_client,
                primary_model=primary_opts.get("model", DEFAULT_GROQ_MODEL),
                fallback_models=get_fallback_models(),
                temperature=0.0,
                groq_api_key=os.getenv("GROQ_API_KEY"),
                chat_groq_cls=ChatGroq,
            )
        return self._llm

    def get_memories(self, user_id: str) -> dict[str, str]:
        """Fetch all stored memories for a user."""
        if not user_id:
            return {}

        with get_db_session() as session:
            records = session.query(UserMemoryRecord).filter_by(user_id=user_id).all()
            return {r.key: r.value for r in records}

    def save_memory(
        self,
        user_id: str,
        key: str,
        value: str,
        source_conversation_id: Optional[str] = None,
    ) -> None:
        """Upsert a single durable memory for a user."""
        if not user_id or not key or not value:
            return

        with get_db_session() as session:
            record = (
                session.query(UserMemoryRecord)
                .filter_by(user_id=user_id, key=key)
                .first()
            )
            if record:
                record.value = value
                record.source_conversation_id = source_conversation_id
                record.updated_at = datetime.now(timezone.utc)
            else:
                session.add(
                    UserMemoryRecord(
                        user_id=user_id,
                        key=key,
                        value=value,
                        source_conversation_id=source_conversation_id,
                        updated_at=datetime.now(timezone.utc),
                    )
                )

    def format_memory_prompt(self, user_id: str) -> str:
        """Format memories into an agent prompt section."""
        memories = self.get_memories(user_id)
        if not memories:
            return ""

        lines = [
            "CUSTOMER PROFILE & DURABLE MEMORY (Remembered from previous interactions):",
        ]
        for k, v in memories.items():
            lines.append(f"- {k}: {v}")
        lines.append(
            "Use this customer context to personalize your assistance without asking the customer to re-explain known facts."
        )
        return "\n".join(lines)

    def extract_and_save_facts(
        self,
        user_id: str,
        conversation_id: str,
        messages: list[Any],
    ) -> dict[str, str]:
        """Extract durable facts from conversation and persist them."""
        if not user_id or not messages:
            return {}

        # Format conversation transcript
        transcript_lines = []
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if role in ["user", "assistant"]:
                transcript_lines.append(f"{role.upper()}: {content}")

        transcript = "\n".join(transcript_lines[-8:])  # Last 8 turns

        prompt = f"""You are a customer intelligence agent. Analyze this customer support conversation transcript and extract durable, long-term facts about the user.
Do NOT extract transient questions or temporary troubleshooting steps. ONLY extract durable facts that will remain true in future sessions.

CONVERSATION TRANSCRIPT:
{transcript}

Respond ONLY with valid JSON in this exact structure:
{{
  "plan_tier": "<Starter|Pro|Enterprise|null>",
  "integrations_used": ["<integration1>", "<integration2>"],
  "recurring_issues": ["<issue1>"],
  "preferences": ["<preference1>"],
  "custom_facts": {{
    "<key>": "<value>"
  }}
}}"""

        try:
            llm = self._get_llm()
            resp = llm.invoke(prompt)
            raw = resp.content.strip()

            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            data = json.loads(raw)
            facts = DurableFacts(**data)

            # Persist extracted durable facts
            if facts.plan_tier:
                self.save_memory(user_id, "plan_tier", facts.plan_tier, conversation_id)

            if facts.integrations_used:
                existing = self.get_memories(user_id).get("integrations_used", "")
                existing_set = set(x.strip() for x in existing.split(",") if x.strip())
                new_set = existing_set | set(facts.integrations_used)
                self.save_memory(user_id, "integrations_used", ", ".join(sorted(new_set)), conversation_id)

            if facts.recurring_issues:
                self.save_memory(user_id, "recurring_issues", "; ".join(facts.recurring_issues), conversation_id)

            if facts.preferences:
                self.save_memory(user_id, "preferences", "; ".join(facts.preferences), conversation_id)

            for k, v in facts.custom_facts.items():
                if v:
                    self.save_memory(user_id, k, str(v), conversation_id)

            logger.info("user_memory_extracted", user_id=user_id, conversation_id=conversation_id)
            return self.get_memories(user_id)

        except Exception as e:
            logger.warning("user_memory_extraction_failed", error=str(e), user_id=user_id)
            return self.get_memories(user_id)

    def clear_user_memories(self, user_id: str) -> None:
        """Clear all stored memories for a user (primarily for tests)."""
        with get_db_session() as session:
            session.query(UserMemoryRecord).filter_by(user_id=user_id).delete()
