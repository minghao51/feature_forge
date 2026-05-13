"""Unit tests for LLM provider factory."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from feature_forge.config import LLMConfig, RetryConfig
from feature_forge.exceptions import LLMError
from feature_forge.llm.factory import _infer_provider, create_llm_client
from feature_forge.llm.providers.anthropic import AnthropicProvider
from feature_forge.llm.providers.deepseek import DeepSeekProvider
from feature_forge.llm.providers.openai import OpenAIProvider


class TestInferProvider:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("deepseek-chat", "deepseek"),
            ("deepseek-reasoner", "deepseek"),
            ("gpt-4", "openai"),
            ("gpt-3.5-turbo", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("o4-mini", "openai"),
            ("claude-3-opus", "anthropic"),
            ("claude-3.5-sonnet", "anthropic"),
            ("llama-3-70b", "litellm"),
            ("gemini-pro", "litellm"),
            ("something-unknown", "litellm"),
        ],
    )
    def test_model_prefix_mapping(self, model: str, expected: str) -> None:
        assert _infer_provider(model) == expected


class TestCreateLLMClient:
    @pytest.mark.llm
    def test_explicit_deepseek_provider(self) -> None:
        config = LLMConfig(model="deepseek-chat", provider="deepseek", api_key=SecretStr("sk-test"))
        client = create_llm_client(config)
        assert isinstance(client, DeepSeekProvider)
        assert client.model == "deepseek-chat"

    @pytest.mark.llm
    def test_auto_infer_deepseek(self) -> None:
        config = LLMConfig(model="deepseek-chat", provider="auto", api_key=SecretStr("sk-test"))
        client = create_llm_client(config)
        assert isinstance(client, DeepSeekProvider)

    @pytest.mark.llm
    def test_auto_infer_openai(self) -> None:
        config = LLMConfig(model="gpt-4", provider="auto", api_key=SecretStr("sk-test"))
        client = create_llm_client(config)
        assert isinstance(client, OpenAIProvider)
        assert client.model == "gpt-4"

    @pytest.mark.llm
    def test_auto_infer_anthropic(self) -> None:
        config = LLMConfig(model="claude-3-opus", provider="auto", api_key=SecretStr("sk-test"))
        client = create_llm_client(config)
        assert isinstance(client, AnthropicProvider)

    @pytest.mark.llm
    def test_auto_infer_unknown_falls_back_to_litellm(self) -> None:
        config = LLMConfig(model="llama-3-70b", provider="auto", api_key=SecretStr("sk-test"))
        with pytest.raises(LLMError, match="litellm is not installed"):
            create_llm_client(config)

    @pytest.mark.llm
    def test_unknown_provider_falls_back_to_litellm(self) -> None:
        config = LLMConfig(model="anything", provider="litellm", api_key=SecretStr("sk-test"))
        with pytest.raises(LLMError, match="litellm is not installed"):
            create_llm_client(config)

    @pytest.mark.llm
    def test_retry_config_attached(self) -> None:
        config = LLMConfig(model="deepseek-chat", provider="deepseek", api_key=SecretStr("sk-test"))
        retry = RetryConfig(max_retries=5, backoff_base=2.0)
        client = create_llm_client(config, retry_config=retry)
        assert client._retry_config is not None
        assert client._retry_config.max_retries == 5
        assert client._retry_config.backoff_base == 2.0

    @pytest.mark.llm
    def test_no_retry_config_when_not_provided(self) -> None:
        config = LLMConfig(model="deepseek-chat", provider="deepseek", api_key=SecretStr("sk-test"))
        client = create_llm_client(config)
        assert client._retry_config is None

    @pytest.mark.llm
    def test_secret_str_api_key(self) -> None:
        secret = SecretStr("sk-secret-value")
        config = LLMConfig(model="deepseek-chat", provider="deepseek", api_key=secret)
        client = create_llm_client(config)
        assert client.api_key == "sk-secret-value"
