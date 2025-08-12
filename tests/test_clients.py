import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.clients.gemini_client import GeminiClient
from app.clients.ollama_client import OllamaClient
from app.core.exceptions import LLMClientError, LLMParseError, LLMTimeoutError
from app.models.classification import GeminiConfig, OllamaConfig


class TestOllamaClient:
    @pytest.fixture
    def client(self):
        config = OllamaConfig(base_url="http://localhost:11434", model="qwen2.5:1.5b")
        mock_prompt_provider = Mock()
        return OllamaClient(config, mock_prompt_provider)

    def test_client_initialization_custom_params(self):
        config = OllamaConfig(base_url="http://custom:8080", model="custom-model")
        mock_prompt_provider = Mock()
        client = OllamaClient(config, mock_prompt_provider)
        assert client.ollama_config.model == "custom-model"
        assert client.ollama_config.base_url == "http://custom:8080"

    @pytest.mark.asyncio
    async def test_classify_success(self, client):
        mock_response_data = {
            "response": '{"category": "transport", "reasoning": "Gas station purchase"}'
        }

        with patch.object(client, "_make_classification_request") as mock_request:
            mock_request.return_value = mock_response_data

            with patch("app.core.prompt_loader.PromptLoader") as mock_loader_class:
                mock_loader = Mock()
                mock_loader.get_formatted_prompt.return_value = "Classify: Gas station"
                mock_loader.get_prompt.side_effect = [
                    "System prompt",
                    "JSON instruction",
                ]
                mock_loader_class.return_value = mock_loader

                async with client:
                    result = await client.classify(
                        payment_text="Shell gas station",
                        categories=["transport", "other"],
                    )

                    assert result.category == "transport"
                    assert result.reasoning == "Gas station purchase"

    @pytest.mark.asyncio
    async def test_classify_timeout_status(self, client):
        with patch.object(client, "_make_classification_request") as mock_request:
            from app.core.exceptions import LLMTimeoutError

            mock_request.side_effect = LLMTimeoutError(
                "Ollama timed out (status 408)", "test-id", "qwen2.5:1.5b"
            )

            with patch("app.core.prompt_loader.PromptLoader"):
                async with client:
                    with pytest.raises(LLMTimeoutError, match="Ollama timed out"):
                        await client.classify("test", ["category"])

    @pytest.mark.asyncio
    async def test_classify_invalid_json(self, client):
        mock_response_data = {"response": "invalid json{"}

        with patch.object(client, "_make_classification_request") as mock_request:
            mock_request.return_value = mock_response_data

            with patch("app.core.prompt_loader.PromptLoader"):
                async with client:
                    with pytest.raises(
                        LLMParseError, match="Invalid JSON response from Ollama"
                    ):
                        await client.classify("test", ["category"])


class TestGeminiClient:
    def test_client_initialization_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMClientError, match="Google API key is required"):
                config = GeminiConfig(api_key=None)
                mock_prompt_provider = Mock()
                GeminiClient(config, mock_prompt_provider)

    def test_client_initialization_with_api_key(self):
        with patch("google.generativeai.configure") as mock_configure:
            with patch("google.generativeai.GenerativeModel") as mock_model:
                config = GeminiConfig(api_key="test-key")
                mock_prompt_provider = Mock()
                client = GeminiClient(config, mock_prompt_provider)
                assert client.get_model_name() == "gemini-2.5-flash"
                mock_configure.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_classify_success(self):
        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel") as mock_model_class:
                mock_model = AsyncMock()
                mock_response = Mock()
                mock_response.text = (
                    '{"category": "groceries", "reasoning": "Supermarket purchase"}'
                )
                mock_model.generate_content_async.return_value = mock_response
                mock_model_class.return_value = mock_model

                config = GeminiConfig(api_key="test-key")
                mock_prompt_provider = Mock()
                client = GeminiClient(config, mock_prompt_provider)

                with patch("app.core.prompt_loader.PromptLoader") as mock_loader_class:
                    mock_loader = Mock()
                    mock_loader.get_formatted_prompt.return_value = "Classify: Walmart"
                    mock_loader.get_prompt.return_value = "System prompt"
                    mock_loader_class.return_value = mock_loader

                    result = await client.classify(
                        payment_text="Walmart grocery",
                        categories=["groceries", "other"],
                    )

                    assert result.category == "groceries"
                    assert result.reasoning == "Supermarket purchase"
