from app.clients.llm_client import LLMClient
from app.schemas.classification import ClassificationRequest, PaymentClassification
from app.utils.prompt_loader import PromptLoader


class ClassificationService:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.prompt_loader = PromptLoader()

    async def classify_payment(
        self, request: ClassificationRequest
    ) -> PaymentClassification:
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
