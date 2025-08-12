import json
import os
from typing import Any, List

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.clients.base_client import BaseLLMClient
from app.core.exceptions import LLMClientError, LLMParseError
from app.models.classification import ClassificationResult, OpenAIConfig


class OpenAIClient(BaseLLMClient):
    @staticmethod
    def _is_server_error(e: Exception) -> bool:
        return isinstance(e, APIStatusError) and e.status_code >= 500

    RETRYABLE_EXCEPTIONS = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
    )

    def __init__(self, config: OpenAIConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.openai_config = config
        api_key = self.openai_config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMClientError("OpenAI API key is required.")
        self.client = AsyncOpenAI(api_key=api_key, max_retries=0)

        def retry_condition(retry_state):
            """Custom retry condition that includes server errors"""
            if retry_state.outcome is None:
                return False

            exception = retry_state.outcome.exception()
            if exception is None:
                return False

            # Check for base retryable exceptions
            if isinstance(exception, self.RETRYABLE_EXCEPTIONS):
                return True

            # Check for server errors
            return self._is_server_error(exception)

        self._make_classification_request_with_retry = retry(
            reraise=True,
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(initial=1.0, max=10.0),
            retry=retry_condition,
        )(self._make_classification_request_impl)

    def get_model_name(self) -> str:
        return self.openai_config.model

    async def _make_classification_request(
        self,
        payment_text: str,
        categories: List[str],
        correlation_id: str,
        use_search: bool = False,
    ) -> Any:
        return await self._make_classification_request_with_retry(
            payment_text, categories, correlation_id, use_search
        )

    async def _make_classification_request_impl(
        self,
        payment_text: str,
        categories: List[str],
        correlation_id: str,
        use_search: bool = False,
    ) -> Any:
        system_prompt = self.prompt_provider.get_prompt("system_prompt")
        user_prompt = self.prompt_provider.get_formatted_prompt(
            "classify_user_prompt",
            payment_text=payment_text,
            valid_categories=", ".join(categories),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return await self.client.chat.completions.create(  # type: ignore[call-overload]
            model=self.openai_config.model,
            messages=messages,
            temperature=self.openai_config.temperature,
            max_tokens=self.openai_config.max_tokens,
            response_format={"type": "json_object"},
        )

    def _parse_response(
        self, response: Any, correlation_id: str
    ) -> ClassificationResult:
        try:
            content = response.choices[0].message.content
            parsed_json = json.loads(content)
            self._validate_response_schema(parsed_json)
            return ClassificationResult(
                category=parsed_json["category"],
                reasoning=parsed_json["reasoning"],
                confidence=parsed_json.get("confidence"),
                metadata={
                    "raw_response": parsed_json,
                    "usage": response.usage.model_dump() if response.usage else None,
                },
            )
        except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as e:
            raise LLMParseError(
                f"Invalid response from OpenAI: {getattr(response.choices[0].message, 'content', 'N/A')}",
                correlation_id,
                self.get_model_name(),
            ) from e
