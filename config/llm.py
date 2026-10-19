"""Shared Groq model selection for support agents and query rewriting."""

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def load_agent_config() -> dict[str, Any]:
    """Load the repository's YAML agent configuration."""
    config_path = Path(__file__).parent / "agents.yaml"
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def groq_model_options(configured_model: str = DEFAULT_GROQ_MODEL) -> dict[str, Any]:
    """Apply an optional deployment override without switching providers."""
    model = os.getenv("GROQ_MODEL", "").strip() or configured_model or DEFAULT_GROQ_MODEL
    options: dict[str, Any] = {"model": model}
    if model.startswith("openai/gpt-oss"):
        # GPT-OSS does not support reasoning_format. Keep only the final answer
        # in responses consumed by the agents' JSON parsers and query rewriting.
        options["model_kwargs"] = {"include_reasoning": False}
    return options
