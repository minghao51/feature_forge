"""Contract and edge-case tests for LLM provider implementations.

Tests provider-specific hooks (_call_api, _extract_content, _extract_usage)
using mocked responses. Does NOT make real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from feature_forge.exceptions import LLMError
from feature_forge.llm.base import LLMClient
from feature_forge.llm.providers.anthropic import AnthropicProvider
from feature_forge.llm.providers.deepseek import DeepSeekProvider
from feature_forge.llm.providers.openai import OpenAIProvider

pytestmark = pytest.mark.contract


# ── Provider ABC contract ────────────────────────────────────────────


class TestAllProvidersContract:
    def test_all_providers_have_required_methods(self):
        required = {"provider_name", "_call_api", "_extract_content", "_extract_usage"}
        for cls in (OpenAIProvider, DeepSeekProvider, AnthropicProvider):
            assert required.issubset(dir(cls)), f"{cls.__name__} missing methods"

    def test_all_providers_subclass_llmclient(self):
        for cls in (OpenAIProvider, DeepSeekProvider, AnthropicProvider):
            assert issubclass(cls, LLMClient)


# ── OpenAI Provider ──────────────────────────────────────────────────


class TestOpenAIProvider:
    def test_missing_api_key_raises(self):
        with pytest.raises(LLMError, match="API key"):
            OpenAIProvider(api_key=None)

    def test_provider_name(self):
        provider = OpenAIProvider(api_key="sk-test")
        assert provider.provider_name == "openai"

    def test_json_mode_kwargs(self):
        provider = OpenAIProvider(api_key="sk-test")
        assert provider._json_mode_kwargs() == {"response_format": {"type": "json_object"}}

    def test_extract_content_returns_text(self):
        provider = OpenAIProvider(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello world"
        result = provider._extract_content(mock_response)
        assert result == "Hello world"

    def test_extract_content_empty_fallback(self):
        provider = OpenAIProvider(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.choices[0].message.content = None
        result = provider._extract_content(mock_response)
        assert result == ""

    def test_extract_usage_with_values(self):
        provider = OpenAIProvider(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        result = provider._extract_usage(mock_response)
        assert result == (10, 20, 30)

    def test_extract_usage_none(self):
        provider = OpenAIProvider(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.usage = None
        result = provider._extract_usage(mock_response)
        assert result == (0, 0, 0)


# ── DeepSeek Provider ────────────────────────────────────────────────


class TestDeepSeekProvider:
    def test_provider_name(self):
        provider = DeepSeekProvider(api_key="sk-test")
        assert provider.provider_name == "deepseek"

    def test_default_base_url(self):
        provider = DeepSeekProvider(api_key="sk-test")
        assert provider.base_url == "https://api.deepseek.com"

    def test_default_model(self):
        provider = DeepSeekProvider(api_key="sk-test")
        assert provider.model == "deepseek-chat"

    def test_thinking_disabled_by_default(self):
        provider = DeepSeekProvider(api_key="sk-test")
        assert not provider.thinking_enabled

    def test_thinking_enabled_kwargs(self):
        provider = DeepSeekProvider(api_key="sk-test", thinking_enabled=True)
        assert provider.thinking_enabled

    def test_extract_content_with_text(self):
        provider = DeepSeekProvider(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello DeepSeek"
        result = provider._extract_content(mock_response)
        assert result == "Hello DeepSeek"

    def test_extract_content_none_fallback(self):
        provider = DeepSeekProvider(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.choices[0].message.content = None
        result = provider._extract_content(mock_response)
        assert result == ""

    def test_extract_usage_with_values(self):
        provider = DeepSeekProvider(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 20
        result = provider._extract_usage(mock_response)
        assert result == (5, 15, 20)

    def test_extract_usage_none(self):
        provider = DeepSeekProvider(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.usage = None
        result = provider._extract_usage(mock_response)
        assert result == (0, 0, 0)

    @patch.dict("os.environ", {}, clear=True)
    def test_fallback_to_env_var(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-env-key"}):
            provider = DeepSeekProvider()
            assert provider.get_api_key() == "sk-env-key"


# ── Anthropic Provider ───────────────────────────────────────────────


class TestAnthropicProvider:
    def test_provider_name(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider.provider_name == "anthropic"

    def test_missing_sdk_raises(self):
        with patch("feature_forge.llm.providers.anthropic.AsyncAnthropic", None):
            with pytest.raises(LLMError, match="Anthropic SDK not installed"):
                AnthropicProvider(api_key="sk-ant-test")

    def test_missing_api_key_raises(self):
        with pytest.raises(LLMError, match="API key"):
            AnthropicProvider(api_key=None)

    def test_json_mode_kwargs_empty(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider._json_mode_kwargs() == {}

    def test_extract_content_with_text_blocks(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        mock_response = MagicMock()
        block1 = MagicMock()
        block1.type = "text"
        block1.text = "Hello"
        block2 = MagicMock()
        block2.type = "text"
        block2.text = " Anthropic"
        mock_response.content = [block1, block2]
        result = provider._extract_content(mock_response)
        assert result == "Hello Anthropic"

    def test_extract_content_skips_non_text_blocks(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        mock_response = MagicMock()
        block1 = MagicMock()
        block1.type = "tool_use"
        block2 = MagicMock()
        block2.type = "text"
        block2.text = "only text"
        mock_response.content = [block1, block2]
        result = provider._extract_content(mock_response)
        assert result == "only text"

    def test_extract_content_empty(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        mock_response = MagicMock()
        mock_response.content = []
        result = provider._extract_content(mock_response)
        assert result == ""

    def test_extract_usage_with_values(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        result = provider._extract_usage(mock_response)
        assert result == (10, 20, 30)

    def test_extract_usage_none(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        mock_response = MagicMock()
        mock_response.usage = None
        result = provider._extract_usage(mock_response)
        assert result == (0, 0, 0)

    @pytest.mark.asyncio
    async def test_call_api_splits_system_message(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = MagicMock()
        provider._client = mock_client
        messages = [
            {"role": "system", "content": "You are a helpful AI"},
            {"role": "user", "content": "Hello"},
        ]
        await provider._call_api(messages, temperature=0.5, max_tokens=100)
        mock_client.messages.create.assert_called_once()
        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["system"] == "You are a helpful AI"
        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
