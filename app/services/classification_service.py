import time
from typing import Optional

from app.clients.llm_client import LLMClientManager
from app.core.exceptions import LLMClientError, LLMParseError, LLMTimeoutError
from app.core.logging import log_classification
from app.models.classification import ClassificationRequest, PaymentClassification


class ClassificationService:
    def __init__(self, llm_client_manager: LLMClientManager):
        self.llm_client_manager = llm_client_manager

    async def classify(
        self, classification_request: ClassificationRequest
    ) -> PaymentClassification:
        start_time = time.time()
        error: Optional[Exception] = None
        result = None

        try:
            if classification_request.model_type.value == "local":
                provider_type = "ollama"
            else:
                provider_type = "gemini"

            result = await self.llm_client_manager.classify(
                provider_type=provider_type,
                payment_text=classification_request.payment_text,
                categories=classification_request.categories,
                use_search=classification_request.use_search,
                model_name=classification_request.model_name,
            )

            classification_result = PaymentClassification(
                category=result.category,
                reasoning=result.reasoning,
                search_used=result.metadata.get("search_used", False),
            )

            return classification_result

        except (
            LLMTimeoutError,
            LLMParseError,
            LLMClientError,
            ValueError,
        ) as e:
            error = e
            raise
        except Exception as e:
            error = e
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000
            model_name = result.model_used if result else "unknown"
            if model_name is None:
                model_name = "unknown"

            log_classification(
                model_name=model_name,
                result_category=result.category if result else None,
                duration_ms=duration_ms,
                confidence=result.confidence if result else None,
                error=str(error) if error else None,
            )
