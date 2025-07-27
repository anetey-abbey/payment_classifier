import pytest
from unittest.mock import AsyncMock, Mock, patch
from pydantic import BaseModel

from app.clients.llm_client import LLMClient, LocalLMStudioClient


class MockResponse(BaseModel):
    message: str
    value: int


class TestLocalLMStudioClient:
    @pytest.fixture
    def client(self):
        return LocalLMStudioClient()

    @pytest.fixture
    def mock_openai_client(self):
        return AsyncMock()

    @pytest.fixture
    def mock_response(self):
        mock_choice = Mock()
        mock_choice.message.parsed = MockResponse(message="test", value=42)
        mock_resp = Mock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def test_client_initialization_custom_params(self):
        client = LocalLMStudioClient(
            base_url="http://custom:8080/v1",
            api_key="custom-key",
            model="custom-model"
        )
        assert client.model == "custom-model"

    @pytest.mark.asyncio
    async def test_get_structured_response_success(self, client, mock_openai_client, mock_response):
        with patch.object(client, 'client', mock_openai_client):
            mock_openai_client.beta.chat.completions.parse.return_value = mock_response
            
            result = await client.get_structured_response(
                prompt="test prompt",
                response_model=MockResponse,
                system_prompt="test system"
            )
            
            assert isinstance(result, MockResponse)
            assert result.message == "test"
            assert result.value == 42
            
            mock_openai_client.beta.chat.completions.parse.assert_called_once_with(
                model="qwen3-8b-instruct",
                messages=[
                    {"role": "system", "content": "test system"},
                    {"role": "user", "content": "test prompt"}
                ],
                response_format=MockResponse
            )

    @pytest.mark.asyncio
    async def test_get_structured_response_empty_system_prompt(self, client, mock_openai_client, mock_response):
        with patch.object(client, 'client', mock_openai_client):
            mock_openai_client.beta.chat.completions.parse.return_value = mock_response
            
            result = await client.get_structured_response(
                prompt="test prompt",
                response_model=MockResponse
            )
            
            mock_openai_client.beta.chat.completions.parse.assert_called_once_with(
                model="qwen3-8b-instruct",
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "test prompt"}
                ],
                response_format=MockResponse
            )

    @pytest.mark.asyncio
    async def test_get_structured_response_api_error(self, client, mock_openai_client):
        mock_openai_client.beta.chat.completions.parse.side_effect = Exception("API Error")
        
        with patch.object(client, 'client', mock_openai_client):
            with pytest.raises(Exception, match="API Error"):
                await client.get_structured_response(
                    prompt="test",
                    response_model=MockResponse
                )

    @pytest.mark.asyncio
    async def test_get_structured_response_missing_parsed_attribute(self, client, mock_openai_client):
        mock_choice = Mock()
        del mock_choice.message.parsed
        mock_resp = Mock()
        mock_resp.choices = [mock_choice]
        mock_openai_client.beta.chat.completions.parse.return_value = mock_resp
        
        with patch.object(client, 'client', mock_openai_client):
            with pytest.raises(RuntimeError, match="Failed to get structured response"):
                await client.get_structured_response(
                    prompt="test",
                    response_model=MockResponse
                )

    @pytest.mark.asyncio
    async def test_get_structured_response_with_custom_model(self):
        client = LocalLMStudioClient(model="custom-model")
        mock_openai_client = AsyncMock()
        mock_choice = Mock()
        mock_choice.message.parsed = MockResponse(message="custom", value=100)
        mock_resp = Mock()
        mock_resp.choices = [mock_choice]
        mock_openai_client.beta.chat.completions.parse.return_value = mock_resp
        
        with patch.object(client, 'client', mock_openai_client):
            result = await client.get_structured_response(
                prompt="test",
                response_model=MockResponse
            )
            
            assert result.message == "custom"
            mock_openai_client.beta.chat.completions.parse.assert_called_once_with(
                model="custom-model",
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "test"}
                ],
                response_format=MockResponse
            )


class TestLLMClientABC:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            LLMClient()

    def test_abstract_method_signature(self):
        assert hasattr(LLMClient, 'get_structured_response')
        assert LLMClient.get_structured_response.__isabstractmethod__