"""Offline Groq client configuration regressions for agents and query rewriting."""

from unittest.mock import Mock

import pytest

from agents import base_agent

DEFAULT_MODEL = "openai/gpt-oss-120b"
SYNTHETIC_API_KEY = "synthetic-groq-api-key"


class _ConcreteAgent(base_agent.BaseAgent):
    def process(self, state):
        return state


@pytest.fixture(autouse=True)
def groq_environment(monkeypatch):
    # conftest loads .env; replace relevant values before constructing any client.
    monkeypatch.setenv("GROQ_API_KEY", SYNTHETIC_API_KEY)
    monkeypatch.delenv("GROQ_MODEL", raising=False)


@pytest.fixture
def chat_groq(monkeypatch):
    constructor = Mock(name="ChatGroq")
    monkeypatch.setattr(base_agent, "ChatGroq", constructor)
    return constructor


@pytest.fixture
def llm_config(monkeypatch):
    config = {"model": "configured-custom-model", "temperature": 0.17, "max_tokens": 321}
    monkeypatch.setattr(base_agent, "load_agent_config", lambda: {"llm": config})
    return config


def test_yaml_default_uses_supported_model(chat_groq):
    config = base_agent.load_agent_config()
    assert config["llm"]["model"] == DEFAULT_MODEL

    _ConcreteAgent("test-agent")

    chat_groq.assert_called_once_with(
        model=DEFAULT_MODEL,
        temperature=0.3,
        max_tokens=1024,
        groq_api_key=SYNTHETIC_API_KEY,
        model_kwargs={"include_reasoning": False},
    )


@pytest.mark.parametrize(
    "options, temperature, max_tokens",
    [({}, 0.3, 1024), ({"temperature": 0.17, "max_tokens": 321}, 0.17, 321)],
    ids=["default-options", "configured-options"],
)
def test_missing_model_uses_supported_fallback(
    chat_groq, llm_config, options, temperature, max_tokens
):
    llm_config.clear()
    llm_config.update(options)

    _ConcreteAgent("test-agent")

    chat_groq.assert_called_once_with(
        model=DEFAULT_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=SYNTHETIC_API_KEY,
        model_kwargs={"include_reasoning": False},
    )


@pytest.mark.parametrize(
    "override, expected_model, reasoning_kwargs",
    [
        ("custom-override-model", "custom-override-model", {}),
        (" \tcustom-override-model\n", "custom-override-model", {}),
        (
            " \topenai/gpt-oss-20b\n",
            "openai/gpt-oss-20b",
            {"model_kwargs": {"include_reasoning": False}},
        ),
    ],
    ids=["custom", "trimmed-custom", "trimmed-gpt-oss"],
)
def test_groq_model_overrides_yaml(
    monkeypatch, chat_groq, llm_config, override, expected_model, reasoning_kwargs
):
    llm_config["model"] = DEFAULT_MODEL
    monkeypatch.setenv("GROQ_MODEL", override)

    _ConcreteAgent("test-agent")

    chat_groq.assert_called_once_with(
        model=expected_model,
        temperature=0.17,
        max_tokens=321,
        groq_api_key=SYNTHETIC_API_KEY,
        **reasoning_kwargs,
    )


@pytest.mark.parametrize("override", ["", "   ", "\t\n"], ids=["empty", "spaces", "tabs-newline"])
def test_blank_groq_model_preserves_yaml(monkeypatch, chat_groq, llm_config, override):
    monkeypatch.setenv("GROQ_MODEL", override)

    _ConcreteAgent("test-agent")

    chat_groq.assert_called_once_with(
        model="configured-custom-model",
        temperature=0.17,
        max_tokens=321,
        groq_api_key=SYNTHETIC_API_KEY,
    )


@pytest.mark.parametrize(
    "model, reasoning_kwargs",
    [
        ("openai/gpt-oss-120b", {"model_kwargs": {"include_reasoning": False}}),
        ("openai/gpt-oss-20b", {"model_kwargs": {"include_reasoning": False}}),
        ("configured-custom-model", {}),
        ("other/openai/gpt-oss-120b", {}),
    ],
    ids=["gpt-oss-120b", "gpt-oss-20b", "custom", "non-prefix-match"],
)
def test_reasoning_disabled_only_for_gpt_oss(chat_groq, llm_config, model, reasoning_kwargs):
    llm_config["model"] = model

    _ConcreteAgent("test-agent")

    chat_groq.assert_called_once_with(
        model=model,
        temperature=0.17,
        max_tokens=321,
        groq_api_key=SYNTHETIC_API_KEY,
        **reasoning_kwargs,
    )


@pytest.mark.parametrize(
    "yaml_model, explicit_model, override, expected_model, reasoning_kwargs",
    [
        pytest.param(
            DEFAULT_MODEL,
            None,
            None,
            DEFAULT_MODEL,
            {"model_kwargs": {"include_reasoning": False}},
            id="yaml-default",
        ),
        pytest.param(
            "yaml-custom-model",
            None,
            None,
            "yaml-custom-model",
            {},
            id="configured-yaml-model",
        ),
        pytest.param(
            DEFAULT_MODEL,
            None,
            " \topenai/gpt-oss-20b\n",
            "openai/gpt-oss-20b",
            {"model_kwargs": {"include_reasoning": False}},
            id="env-gpt-oss-overrides-yaml",
        ),
        pytest.param(
            DEFAULT_MODEL,
            None,
            " custom-override-model ",
            "custom-override-model",
            {},
            id="env-custom-overrides-yaml",
        ),
        pytest.param(
            None,
            None,
            None,
            DEFAULT_MODEL,
            {"model_kwargs": {"include_reasoning": False}},
            id="missing-yaml-model-fallback",
        ),
        pytest.param(
            DEFAULT_MODEL,
            "explicit-custom-model",
            None,
            "explicit-custom-model",
            {},
            id="explicit-custom-overrides-yaml",
        ),
        pytest.param(
            DEFAULT_MODEL,
            "explicit-custom-model",
            " openai/gpt-oss-20b ",
            "openai/gpt-oss-20b",
            {"model_kwargs": {"include_reasoning": False}},
            id="env-gpt-oss-overrides-explicit",
        ),
        pytest.param(
            DEFAULT_MODEL,
            "openai/gpt-oss-20b",
            "custom-override-model",
            "custom-override-model",
            {},
            id="env-custom-overrides-explicit",
        ),
    ],
)
def test_rag_model_selection(
    monkeypatch, yaml_model, explicit_model, override, expected_model, reasoning_kwargs
):
    # Keep retrieval imports local so agent cases can run without retrieval dependencies.
    from retrieval import pipeline

    config = {"llm": {"temperature": 0.17, "max_tokens": 321}}
    if yaml_model is not None:
        config["llm"]["model"] = yaml_model
    # The old pipeline has no loader symbol; allow it so failures reach the constructor.
    monkeypatch.setattr(pipeline, "load_agent_config", Mock(return_value=config), raising=False)
    chat_groq = Mock(name="ChatGroq")
    monkeypatch.setattr(pipeline, "ChatGroq", chat_groq)
    monkeypatch.setattr(pipeline, "HybridRetriever", Mock(name="HybridRetriever"))
    if override is not None:
        monkeypatch.setenv("GROQ_MODEL", override)

    kwargs = {} if explicit_model is None else {"llm_model": explicit_model}
    pipeline.RAGPipeline(
        vector_store=Mock(name="vector_store"),
        bm25_store=Mock(name="bm25_store"),
        reranker=Mock(name="reranker"),
        **kwargs,
    )

    chat_groq.assert_called_once_with(
        model=expected_model,
        temperature=0,
        groq_api_key=SYNTHETIC_API_KEY,
        **reasoning_kwargs,
    )
