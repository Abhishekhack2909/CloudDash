"""Shared Groq model configuration, validation, and fallback wrapper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from langchain_groq import ChatGroq
import yaml

from observability.logger import get_logger

logger = get_logger(__name__)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_FALLBACK_MODELS = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

_RUNTIME_STATUS: dict[str, Any] = {
    "primary_model": DEFAULT_GROQ_MODEL,
    "active_model": DEFAULT_GROQ_MODEL,
    "fallback_used": False,
    "fallback_models": list(DEFAULT_GROQ_FALLBACK_MODELS),
    "available_models": [],
    "last_error": None,
}


def load_agent_config() -> dict[str, Any]:
    """Load the repository's YAML agent configuration."""
    config_path = Path(__file__).parent / "agents.yaml"
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def get_primary_model(configured_model: str | None = None) -> str:
    """Get the primary Groq model name from env or config with default fallback."""
    env_model = os.getenv("GROQ_MODEL", "").strip()
    if env_model:
        return env_model
    if configured_model and configured_model.strip():
        return configured_model.strip()
    yaml_config = load_agent_config().get("llm", {})
    yaml_model = str(yaml_config.get("model", "")).strip()
    return yaml_model or DEFAULT_GROQ_MODEL


def get_fallback_models(configured_fallbacks: list[str] | None = None) -> list[str]:
    """Get fallback Groq models from env or config."""
    env_fallbacks = os.getenv("GROQ_FALLBACK_MODELS", "").strip()
    if env_fallbacks:
        models = [m.strip() for m in env_fallbacks.split(",") if m.strip()]
        if models:
            return models
    if configured_fallbacks:
        return [m.strip() for m in configured_fallbacks if m.strip()]
    yaml_config = load_agent_config().get("llm", {})
    yaml_fallbacks = yaml_config.get("fallback_models")
    if yaml_fallbacks and isinstance(yaml_fallbacks, list):
        return [str(m).strip() for m in yaml_fallbacks if str(m).strip()]
    return list(DEFAULT_GROQ_FALLBACK_MODELS)


def groq_model_options(configured_model: str = DEFAULT_GROQ_MODEL) -> dict[str, Any]:
    """Apply model options (e.g. disabling reasoning tokens for GPT-OSS)."""
    model = get_primary_model(configured_model)
    options: dict[str, Any] = {"model": model}
    if model.startswith("openai/gpt-oss"):
        # GPT-OSS does not support reasoning_format; disable reasoning output
        # to ensure downstream JSON parsers receive only pure JSON answers.
        options["model_kwargs"] = {"include_reasoning": False}
    return options


def record_active_model(model: str, fallback_used: bool = False):
    """Record which model is currently being used at runtime."""
    _RUNTIME_STATUS["active_model"] = model
    if fallback_used:
        _RUNTIME_STATUS["fallback_used"] = True


def get_runtime_model_status() -> dict[str, Any]:
    """Get the runtime model status dictionary."""
    primary = get_primary_model()
    _RUNTIME_STATUS["primary_model"] = primary
    _RUNTIME_STATUS["fallback_models"] = get_fallback_models()
    return dict(_RUNTIME_STATUS)


def check_groq_models(api_key: str | None = None) -> dict[str, Any]:
    """Query Groq list-models endpoint at startup, log available models, and validate configuration."""
    key = api_key or os.getenv("GROQ_API_KEY")
    primary = get_primary_model()
    fallbacks = get_fallback_models()
    _RUNTIME_STATUS["primary_model"] = primary
    _RUNTIME_STATUS["fallback_models"] = fallbacks
    _RUNTIME_STATUS["active_model"] = primary

    if not key:
        logger.warning("groq_check_skipped", reason="GROQ_API_KEY not set")
        return get_runtime_model_status()

    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )

        if resp.status_code == 200:
            data = resp.json().get("data", [])
            available = [m.get("id") for m in data if isinstance(m, dict) and "id" in m]
            _RUNTIME_STATUS["available_models"] = available
            logger.info("groq_models_listed", available_count=len(available), models=available)

            if primary in available:
                _RUNTIME_STATUS["active_model"] = primary
                _RUNTIME_STATUS["fallback_used"] = False
                return get_runtime_model_status()

            # Primary not available; search fallbacks
            for fb in fallbacks:
                if fb in available:
                    logger.warning(
                        "groq_primary_unavailable_using_fallback",
                        configured_primary=primary,
                        active_fallback=fb,
                        available_models=available,
                    )
                    _RUNTIME_STATUS["active_model"] = fb
                    _RUNTIME_STATUS["fallback_used"] = True
                    return get_runtime_model_status()

            # Neither primary nor fallbacks available
            err_msg = (
                f"Configured Groq model '{primary}' and fallbacks {fallbacks} are not available "
                f"in your Groq account. Available models: {available}. "
                f"Please update GROQ_MODEL in your environment settings."
            )
            logger.error("groq_model_not_found_on_startup", error=err_msg)
            raise ValueError(err_msg)
        else:
            logger.warning(
                "groq_models_endpoint_error",
                status_code=resp.status_code,
                response_preview=resp.text[:200],
            )
    except ValueError:
        raise
    except Exception as e:
        logger.warning("groq_models_check_failed", error=str(e))

    return get_runtime_model_status()


class FallbackChatGroq:
    """Resilient ChatGroq wrapper that transparently falls back on model errors, rate limits, or 5xx."""

    def __init__(
        self,
        primary_client: Any,
        primary_model: str,
        fallback_models: list[str] | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        groq_api_key: str | None = None,
        chat_groq_cls: Any = ChatGroq,
    ):
        self._primary_client = primary_client
        self._primary_model = primary_model
        raw_fallbacks = fallback_models or get_fallback_models()
        self._fallback_models = [m for m in raw_fallbacks if m != primary_model]
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._groq_api_key = groq_api_key
        self._chat_groq_cls = chat_groq_cls

    def _is_retryable_error(self, exc: Exception) -> bool:
        err_msg = str(exc).lower()
        if "model_not_found" in err_msg or "404" in err_msg or "does not exist" in err_msg:
            return True
        if "429" in err_msg or "rate_limit" in err_msg or "rate limit" in err_msg:
            return True
        if any(code in err_msg for code in ["500", "502", "503", "504", "server_error"]):
            return True
        status_code = getattr(exc, "status_code", None)
        if status_code in (404, 429, 500, 502, 503, 504):
            return True
        return False

    def _build_client_for_model(self, model: str) -> Any:
        opts = groq_model_options(model)
        client_kwargs: dict[str, Any] = {
            **opts,
            "temperature": self._temperature,
            "groq_api_key": self._groq_api_key or os.getenv("GROQ_API_KEY"),
        }
        if self._max_tokens is not None:
            client_kwargs["max_tokens"] = self._max_tokens
        return self._chat_groq_cls(**client_kwargs)

    def invoke(self, input_data: Any, *args, **kwargs) -> Any:
        models_to_try = [self._primary_model] + self._fallback_models
        last_exception = None

        for idx, model in enumerate(models_to_try):
            try:
                if idx == 0 and self._primary_client is not None:
                    client = self._primary_client
                else:
                    client = self._build_client_for_model(model)

                response = client.invoke(input_data, *args, **kwargs)
                record_active_model(model, fallback_used=(idx > 0))
                return response
            except Exception as e:
                last_exception = e
                if not self._is_retryable_error(e) or idx == len(models_to_try) - 1:
                    logger.error("llm_invocation_failed", model=model, error=str(e))
                    raise
                next_model = models_to_try[idx + 1]
                logger.warning(
                    "llm_model_fallback",
                    failed_model=model,
                    fallback_model=next_model,
                    error=str(e),
                )
                record_active_model(next_model, fallback_used=True)

        if last_exception:
            raise last_exception

    def bind_tools(self, tools: list[Any], *args, **kwargs) -> Any:
        return FallbackChatGroqWithTools(
            fallback_wrapper=self,
            tools=tools,
            bind_args=args,
            bind_kwargs=kwargs,
        )


class FallbackChatGroqWithTools:
    """Wrapper that applies tool binding across primary and fallback Groq models."""

    def __init__(self, fallback_wrapper: FallbackChatGroq, tools: list[Any], bind_args: tuple, bind_kwargs: dict):
        self._wrapper = fallback_wrapper
        self._tools = tools
        self._bind_args = bind_args
        self._bind_kwargs = bind_kwargs

    def invoke(self, input_data: Any, *args, **kwargs) -> Any:
        models_to_try = [self._wrapper._primary_model] + self._wrapper._fallback_models
        last_exception = None

        for idx, model in enumerate(models_to_try):
            try:
                if idx == 0 and self._wrapper._primary_client is not None:
                    raw_client = self._wrapper._primary_client
                else:
                    raw_client = self._wrapper._build_client_for_model(model)

                client = raw_client.bind_tools(self._tools, *self._bind_args, **self._bind_kwargs)
                response = client.invoke(input_data, *args, **kwargs)
                record_active_model(model, fallback_used=(idx > 0))
                return response
            except Exception as e:
                last_exception = e
                if not self._wrapper._is_retryable_error(e) or idx == len(models_to_try) - 1:
                    logger.error("llm_tool_invocation_failed", model=model, error=str(e))
                    raise
                next_model = models_to_try[idx + 1]
                logger.warning(
                    "llm_model_fallback_tools",
                    failed_model=model,
                    fallback_model=next_model,
                    error=str(e),
                )
                record_active_model(next_model, fallback_used=True)

        if last_exception:
            raise last_exception



    async def ainvoke(self, input_data: Any, *args, **kwargs) -> Any:
        models_to_try = [self._primary_model] + self._fallback_models
        last_exception = None

        for idx, model in enumerate(models_to_try):
            try:
                if idx == 0 and self._primary_client is not None:
                    client = self._primary_client
                else:
                    client = self._build_client_for_model(model)

                if hasattr(client, "ainvoke"):
                    response = await client.ainvoke(input_data, *args, **kwargs)
                else:
                    response = client.invoke(input_data, *args, **kwargs)
                record_active_model(model, fallback_used=(idx > 0))
                return response
            except Exception as e:
                last_exception = e
                if not self._is_retryable_error(e) or idx == len(models_to_try) - 1:
                    logger.error("llm_async_invocation_failed", model=model, error=str(e))
                    raise
                next_model = models_to_try[idx + 1]
                logger.warning(
                    "llm_model_fallback",
                    failed_model=model,
                    fallback_model=next_model,
                    error=str(e),
                )
                record_active_model(next_model, fallback_used=True)

        if last_exception:
            raise last_exception


