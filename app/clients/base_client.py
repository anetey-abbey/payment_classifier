import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.core.exceptions import LLMClientError, LLMValidationError
from app.core.protocols import MetricsCollector, PromptProvider, StructuredLogger
from app.models.classification import BaseLLMConfig, ClassificationResult


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


class DefaultMetricsCollector:
    def record_request_duration(self, *args, **kwargs):
        pass

    def increment_counter(self, *args, **kwargs):
        pass


class BaseLLMClient(ABC):
    def __init__(
        self,
        config: BaseLLMConfig,
        prompt_provider: PromptProvider,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.config = config
        self.prompt_provider = prompt_provider
        self.logger = logger or DefaultStructuredLogger()
        self.metrics = metrics or DefaultMetricsCollector()
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def classify(
        self,
        payment_text: str,
        categories: List[str],
        correlation_id: Optional[str] = None,
        use_search: bool = False,
    ) -> ClassificationResult:
        correlation_id = correlation_id or str(uuid.uuid4())
        start_time = time.monotonic()
        async with self._semaphore:
            try:
                self._validate_inputs(payment_text, categories)
                if self.config.enable_request_logging:
                    self.logger.info(
                        "LLM request",
                        correlation_id=correlation_id,
                        model=self.get_model_name(),
                        use_search=use_search,
                    )

                response = await self._make_classification_request(
                    payment_text, categories, correlation_id, use_search
                )

                result = self._parse_response(response, correlation_id)

                duration_ms = (time.monotonic() - start_time) * 1000
                result.correlation_id = correlation_id
                result.model_used = self.get_model_name()
                result.processing_time_ms = round(duration_ms, 2)

                if self.config.enable_response_logging:
                    self.logger.info(
                        "LLM response",
                        correlation_id=correlation_id,
                        category=result.category,
                        duration_ms=result.processing_time_ms,
                    )
                self._record_metrics(duration_ms, True)
                return result
            except Exception as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                is_known_error = isinstance(e, (LLMClientError, LLMValidationError))
                error_type = type(e).__name__ if is_known_error else "UnexpectedError"

                self.logger.error(
                    "LLM classification failed",
                    correlation_id=correlation_id,
                    model=self.get_model_name(),
                    error=str(e),
                    error_type=error_type,
                    duration_ms=round(duration_ms, 2),
                )
                self._record_metrics(duration_ms, False, error_type)

                if is_known_error:
                    raise
                raise LLMClientError(
                    f"Unexpected error in {self.__class__.__name__}",
                    correlation_id,
                    self.get_model_name(),
                ) from e

    def _validate_inputs(self, payment_text: str, categories: List[str]) -> None:
        if not payment_text or not payment_text.strip():
            raise LLMValidationError("payment_text cannot be empty")
        if not categories:
            raise LLMValidationError("categories list cannot be empty")
        if len(categories) > self.config.max_categories:
            raise LLMValidationError(
                f"Too many categories (max {self.config.max_categories})"
            )
        if len(payment_text) > self.config.max_payment_text_length:
            raise LLMValidationError(
                f"payment_text too long (max {self.config.max_payment_text_length} chars)"
            )

    def _validate_response_schema(self, response_data: Dict[str, Any]) -> None:
        if not all(k in response_data for k in ["category", "reasoning"]):
            raise LLMClientError(
                f"Missing required fields in response: {response_data}",
                model=self.get_model_name(),
            )
        if not isinstance(response_data["category"], str) or not isinstance(
            response_data["reasoning"], str
        ):
            raise LLMClientError(
                "Invalid types for category/reasoning in response",
                model=self.get_model_name(),
            )

    def _record_metrics(
        self,
        duration_ms: float,
        success: bool,
        error_type: Optional[str] = None,
    ) -> None:
        self.metrics.record_request_duration(
            self.get_model_name(), duration_ms, success
        )
        tags = {"model": self.get_model_name(), "success": str(success).lower()}
        if error_type:
            tags["error_type"] = error_type
        self.metrics.increment_counter("llm_requests_total", tags)

    @abstractmethod
    async def _make_classification_request(
        self,
        payment_text: str,
        categories: List[str],
        correlation_id: str,
        use_search: bool = False,
    ) -> Any: ...

    @abstractmethod
    def _parse_response(
        self, response: Any, correlation_id: str
    ) -> ClassificationResult: ...

    @abstractmethod
    def get_model_name(self) -> str: ...

    async def _setup(self) -> None:
        pass

    async def _cleanup(self) -> None:
        pass

    async def __aenter__(self):
        await self._setup()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        await self._cleanup()
