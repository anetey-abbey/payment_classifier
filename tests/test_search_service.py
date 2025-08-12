import asyncio
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from app.services.search_service import GoogleSearchService


@pytest.fixture
def search_service():
    return GoogleSearchService(
        api_key="test_key",
        search_engine_id="test_engine_id",
        timeout=5.0,
        max_retries=2,
    )


@pytest.fixture
def mock_response_data():
    return {
        "items": [
            {
                "title": "Test Payment Processor",
                "snippet": "A reliable payment processing service",
                "link": "https://example.com/payment",
            },
            {
                "title": "Payment Gateway Solutions",
                "snippet": "Complete payment solutions for businesses",
                "link": "https://example.com/gateway",
            },
        ]
    }


class TestGoogleSearchService:
    def test_init_with_valid_credentials(self):
        service = GoogleSearchService(
            api_key="valid_key", search_engine_id="valid_engine_id"
        )
        assert service.api_key == "valid_key"
        assert service.search_engine_id == "valid_engine_id"

    def test_init_with_invalid_credentials(self):
        with pytest.raises(
            ValueError, match="Google API key and search engine ID are required"
        ):
            GoogleSearchService(api_key="", search_engine_id="test_engine_id")

        with pytest.raises(
            ValueError, match="Google API key and search engine ID are required"
        ):
            GoogleSearchService(api_key="test_key", search_engine_id="")

    @pytest.mark.asyncio
    async def test_search_without_session_raises_error(self, search_service):
        with pytest.raises(Exception):
            await search_service.search("test query")

    @pytest.mark.asyncio
    async def test_search_with_empty_query_returns_empty_list(self, search_service):
        async with search_service:
            result = await search_service.search("")
            assert result == []

            result = await search_service.search("   ")
            assert result == []

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_successful_search(
        self, mock_get, search_service, mock_response_data
    ):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.raise_for_status = Mock()
        mock_get.return_value.__aenter__.return_value = mock_response

        async with search_service:
            results = await search_service.search("payment processor", num_results=2)

        assert len(results) == 2
        assert results[0]["title"] == "Test Payment Processor"
        assert results[0]["snippet"] == "A reliable payment processing service"
        assert results[0]["link"] == "https://example.com/payment"

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://www.googleapis.com/customsearch/v1"
        assert call_args[1]["params"]["q"] == "payment processor"
        assert call_args[1]["params"]["num"] == 2

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_search_with_rate_limit_error(self, mock_get, search_service):
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_get.return_value.__aenter__.return_value = mock_response

        async with search_service:
            results = await search_service.search("test query")

        assert results == []

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_search_with_timeout_error(self, mock_get, search_service):
        mock_response = AsyncMock()
        mock_response.status = 408
        mock_get.return_value.__aenter__.return_value = mock_response

        async with search_service:
            results = await search_service.search("test query")

        assert results == []

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_search_with_no_results(self, mock_get, search_service):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})
        mock_response.raise_for_status = Mock()
        mock_get.return_value.__aenter__.return_value = mock_response

        async with search_service:
            results = await search_service.search("nonexistent query")

        assert results == []

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_search_limits_num_results(
        self, mock_get, search_service, mock_response_data
    ):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.raise_for_status = Mock()
        mock_get.return_value.__aenter__.return_value = mock_response

        async with search_service:
            await search_service.search("test query", num_results=15)

        call_args = mock_get.call_args
        assert call_args[1]["params"]["num"] == 10  # Should be limited to 10

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_search_includes_correlation_id(
        self, mock_get, search_service, mock_response_data
    ):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.raise_for_status = Mock()
        mock_get.return_value.__aenter__.return_value = mock_response

        correlation_id = "test-correlation-id"

        async with search_service:
            await search_service.search("test query", correlation_id=correlation_id)

        call_args = mock_get.call_args
        assert call_args[1]["headers"]["X-Correlation-ID"] == correlation_id

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.get")
    async def test_search_handles_missing_fields(self, mock_get, search_service):
        incomplete_data = {
            "items": [
                {
                    "title": "Complete Item",
                    "snippet": "Complete snippet",
                    "link": "https://example.com/complete",
                },
                {
                    "title": "Incomplete Item",
                    # missing snippet and link
                },
            ]
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=incomplete_data)
        mock_response.raise_for_status = Mock()
        mock_get.return_value.__aenter__.return_value = mock_response

        async with search_service:
            results = await search_service.search("test query")

        assert len(results) == 2
        assert results[0]["title"] == "Complete Item"
        assert results[1]["title"] == "Incomplete Item"
        assert results[1]["snippet"] == ""
        assert results[1]["link"] == ""
