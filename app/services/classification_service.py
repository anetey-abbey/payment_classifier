import time

from app.clients.llm_client import LLMClient
from app.schemas.classification import ClassificationRequest, PaymentClassification
from app.utils.logging import log_classification
from app.utils.prompt_loader import PromptLoader


class ClassificationService:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.prompt_loader = PromptLoader()

    async def classify_payment(
        self, request: ClassificationRequest
    ) -> PaymentClassification:
        start_time = time.time()
        error = None
        result = None

        try:
            payment_text = request.payment_text
            valid_categories = ", ".join(request.categories)

            prompt = self.prompt_loader.get_formatted_prompt(
                key="classify_user_prompt",
                payment_text=payment_text,
                valid_categories=valid_categories,
            )
            system_prompt = self.prompt_loader.get_prompt(key="system_prompt")
            result = await self.llm_client.get_structured_response(
                prompt=prompt,
                response_model=PaymentClassification,
                system_prompt=system_prompt,
            )
            return result

        except Exception as e:
            error = str(e)
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000
            log_classification(
                model_name=getattr(self.llm_client, "model", "unknown"),
                result_category=result.category if result else None,
                duration_ms=duration_ms,
                confidence=getattr(result, "confidence", None) if result else None,
                error=error,
            )
