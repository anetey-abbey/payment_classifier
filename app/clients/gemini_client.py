import json
import os
from typing import Any, List

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.clients.base_client import BaseLLMClient
from app.core.exceptions import LLMClientError, LLMParseError
from app.models.classification import ClassificationResult, GeminiConfig


class GeminiClient(BaseLLMClient):
    RETRYABLE_EXCEPTIONS = (
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.InternalServerError,
    )

    def __init__(self, config: GeminiConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.gemini_config = config
        if not self.gemini_config.api_key and not os.getenv("GOOGLE_API_KEY"):
            raise LLMClientError("Google API key is required for Gemini client.")
        api_key = self.gemini_config.api_key or os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.gemini_config.model)
        self._make_classification_request_with_retry = retry(
            reraise=True,
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(initial=1.0, max=10.0),
            retry=retry_if_exception_type(self.RETRYABLE_EXCEPTIONS),
        )(self._make_classification_request_impl)

    def get_model_name(self) -> str:
        return self.gemini_config.model

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
            key="classify_user_prompt",
            payment_text=payment_text,
            valid_categories=", ".join(categories),
        )
        json_schema = {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "reasoning": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["category", "reasoning"],
        }
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=json_schema,
            temperature=self.gemini_config.temperature,
            max_output_tokens=self.gemini_config.max_output_tokens,
        )
        try:
            return await self.client.generate_content_async(
                f"{system_prompt}\n{user_prompt}", generation_config=generation_config
            )
        except (
            genai.types.BlockedPromptException,
            genai.types.StopCandidateException,
        ) as e:
            raise LLMClientError(
                f"Gemini content safety error: {e}",
                correlation_id,
                self.get_model_name(),
            ) from e

    def _parse_response(
        self, response: Any, correlation_id: str
    ) -> ClassificationResult:
        try:
            parsed_json = json.loads(response.text)
            self._validate_response_schema(parsed_json)
            return ClassificationResult(
                category=parsed_json["category"],
                reasoning=parsed_json["reasoning"],
                confidence=parsed_json.get("confidence"),
                metadata={"raw_response": parsed_json},
            )
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            raise LLMParseError(
                f"Invalid response from Gemini: {getattr(response, 'text', 'N/A')}",
                correlation_id,
                self.get_model_name(),
            ) from e
