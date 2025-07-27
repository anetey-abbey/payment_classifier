from abc import ABC, abstractmethod
from typing import Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    @abstractmethod
    async def get_structured_response(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str = "",
    ) -> T:
        pass


class LocalLMStudioClient(LLMClient):
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model: str = "qwen3-8b-instruct",
    ):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    async def get_structured_response(
        self, prompt: str, response_model: Type[T], system_prompt: str = ""
    ) -> T:
        try:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_model,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            raise RuntimeError(f"Failed to get structured response: {e}")