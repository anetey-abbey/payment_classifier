import asyncio
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.exceptions import LLMClientError
from app.core.protocols import StructuredLogger


class DefaultStructuredLogger:
    def _log(self, level: str, message: str, **kwargs):
        print(f"[{level}] {message} | {kwargs}")

    def info(self, message: str, **kwargs: Any):
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log("DEBUG", message, **kwargs)


class GoogleSearchService:
    GOOGLE_SEARCH_API_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(
        self,
        api_key: str,
        search_engine_id: str,
        logger: Optional[StructuredLogger] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
    ):
        if not api_key or not search_engine_id:
            raise ValueError("Google API key and search engine ID are required")

        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.logger = logger or DefaultStructuredLogger()
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

        self._search_with_retry = retry(
            reraise=True,
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(initial=1.0, max=5.0),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        )(self._search_impl)

    async def _setup(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )

    async def _cleanup(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def search(
        self, query: str, num_results: int = 3, correlation_id: str = ""
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            self.logger.warning(
                "Empty search query provided", correlation_id=correlation_id
            )
            return []

        if not self._session:
            raise LLMClientError("Search service not initialized")

        try:
            self.logger.info(
                "Google Search request",
                query=query,
                num_results=num_results,
                correlation_id=correlation_id,
            )

            search_results = await self._search_with_retry(
                query, num_results, correlation_id
            )

            self.logger.info(
                "Google Search response",
                query=query,
                results_count=len(search_results),
                correlation_id=correlation_id,
            )

            return search_results

        except Exception as e:
            self.logger.error(
                "Google Search failed",
                query=query,
                error=str(e),
                correlation_id=correlation_id,
            )
            return []

    async def _search_impl(
        self, query: str, num_results: int, correlation_id: str
    ) -> List[Dict[str, Any]]:
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(num_results, 10),
        }

        if self._session is None:
            raise LLMClientError("Search service not initialized")

        async with self._session.get(
            self.GOOGLE_SEARCH_API_URL,
            params=params,
            headers={"X-Correlation-ID": correlation_id} if correlation_id else {},
        ) as response:
            if response.status == 429:
                raise aiohttp.ClientError("Google Search API rate limited")
            if response.status in (408, 504):
                raise asyncio.TimeoutError("Google Search API timeout")

            response.raise_for_status()
            data = await response.json()

        search_results = []
        if "items" in data:
            for item in data["items"]:
                search_results.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "link": item.get("link", ""),
                    }
                )

        return search_results

    async def __aenter__(self):
        await self._setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup()
