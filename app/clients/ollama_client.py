import json
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.clients.base_client import BaseLLMClient
from app.core.config import load_config
from app.core.exceptions import (
    LLMClientError,
    LLMParseError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.models.classification import ClassificationResult, OllamaConfig
from app.services.search_service import GoogleSearchService


class OllamaClient(BaseLLMClient):
    def __init__(self, config: OllamaConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.ollama_config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._search_service: Optional[GoogleSearchService] = None

        # Initialize search service if credentials are available
        if config.google_api_key and config.google_search_engine_id:
            search_config = load_config().get("google_search", {})
            self._search_service = GoogleSearchService(
                api_key=config.google_api_key,
                search_engine_id=config.google_search_engine_id,
                logger=self.logger,
                timeout=search_config.get("timeout", 10.0),
                max_retries=search_config.get("max_retries", 3),
            )

        # Dynamically apply retry decorator with settings from config
        self._make_classification_request_with_retry = retry(
            reraise=True,
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(initial=1.0, max=10.0),
            retry=retry_if_exception_type(
                (aiohttp.ClientError, LLMRateLimitError, LLMTimeoutError)
            ),
        )(self._make_classification_request_impl)

    def get_model_name(self) -> str:
        return self.ollama_config.model

    async def _setup(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        if self._search_service:
            await self._search_service._setup()

    async def _cleanup(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        if self._search_service:
            await self._search_service._cleanup()

    async def _make_classification_request(
        self,
        payment_text: str,
        categories: List[str],
        correlation_id: str,
        use_search: bool = False,
    ) -> Dict[str, Any]:
        return await self._make_classification_request_with_retry(
            payment_text, categories, correlation_id, use_search
        )

    async def _make_classification_request_impl(
        self,
        payment_text: str,
        categories: List[str],
        correlation_id: str,
        use_search: bool = False,
    ) -> Dict[str, Any]:
        if not self._session:
            raise LLMClientError(
                "Session not initialized. Use async with client.",
                correlation_id,
                self.get_model_name(),
            )

        system_prompt = self.prompt_provider.get_prompt("system_prompt")

        search_results_text = ""
        if use_search and self._search_service:
            try:
                search_results = await self._search_service.search(
                    query=payment_text,
                    num_results=load_config()
                    .get("google_search", {})
                    .get("max_results", 3),
                    correlation_id=correlation_id,
                )
                if search_results:
                    search_results_text = "\n".join(
                        [
                            f"- {result['title']}: {result['snippet']}"
                            for result in search_results
                        ]
                    )
            except Exception as e:
                self.logger.warning(
                    "Search failed, continuing without search results",
                    correlation_id=correlation_id,
                    error=str(e),
                )

        if use_search and search_results_text:
            user_prompt = self.prompt_provider.get_formatted_prompt(
                key="classify_user_prompt_with_search",
                payment_text=payment_text,
                valid_categories=", ".join(categories),
                search_results=search_results_text,
            )
        else:
            user_prompt = self.prompt_provider.get_formatted_prompt(
                key="classify_user_prompt",
                payment_text=payment_text,
                valid_categories=", ".join(categories),
            )
        payload = {
            "model": self.ollama_config.model,
            "prompt": f"{system_prompt}\n{user_prompt}",
            "stream": False,
            "format": "json",
            "options": {"temperature": self.ollama_config.temperature},
        }
        url = urljoin(self.ollama_config.base_url, "/api/generate")

        async with self._session.post(
            url,
            json=payload,
            headers={
                "X-Correlation-ID": correlation_id,
            },
        ) as response:
            if response.status == 429:
                raise LLMRateLimitError(
                    "Ollama rate limited", correlation_id, self.get_model_name()
                )
            if response.status in (408, 504):
                raise LLMTimeoutError(
                    f"Ollama timed out (status {response.status})",
                    correlation_id,
                    self.get_model_name(),
                )
            response.raise_for_status()
            response_data = await response.json()

            # Add search usage metadata
            response_data["_search_used"] = use_search and bool(search_results_text)
            return response_data

    def _parse_response(
        self, response: Dict[str, Any], correlation_id: str
    ) -> ClassificationResult:
        try:
            parsed_json = json.loads(response.get("response", "{}"))
            self._validate_response_schema(parsed_json)
            return ClassificationResult(
                category=parsed_json["category"],
                reasoning=parsed_json["reasoning"],
                confidence=parsed_json.get("confidence"),
                metadata={
                    "raw_response": parsed_json,
                    "eval_duration": response.get("eval_duration"),
                    "search_used": response.get("_search_used", False),
                },
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise LLMParseError(
                f"Invalid JSON response from Ollama: {response}",
                correlation_id,
                self.get_model_name(),
            ) from e
