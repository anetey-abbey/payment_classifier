from pathlib import Path

import yaml

from app.clients.llm_client import LLMClient
from app.schemas.classification import PaymentClassification, PaymentData
from app.utils.prompt_loader import PromptLoader


class ClassificationService:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.prompt_loader = PromptLoader()
        self.valid_categories = self._load_valid_categories()

    def _load_valid_categories(self) -> list[str]:
        config_path = Path(__file__).parent.parent.parent / "config" / "categories.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config["payment_categories"]

    async def classify_payment(
        self, payment_data: PaymentData
    ) -> PaymentClassification:
        payment_text = payment_data.payment_text
        prompt = self.prompt_loader.get_formatted_prompt(
            key="classify_user_prompt",
            payment_text=payment_text,
            valid_categories=", ".join(self.valid_categories),
        )
        system_prompt = self.prompt_loader.get_prompt(key="system_prompt")
        result = await self.llm_client.get_structured_response(
            prompt=prompt,
            response_model=PaymentClassification,
            system_prompt=system_prompt,
        )
        return result
