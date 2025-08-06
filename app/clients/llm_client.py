import json
import os
from abc import ABC, abstractmethod
from typing import Type, TypeVar

import aiohttp
import google.generativeai as genai
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.utils.prompt_loader import PromptLoader

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
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str = "",
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
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise RuntimeError("Failed to parse response")
            return parsed
        except Exception as e:
            raise RuntimeError(f"Failed to get structured response: {e}")


class OllamaClient(LLMClient):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompt_loader = PromptLoader()

    def _build_full_prompt(
        self, system_prompt: str, prompt: str, json_instruction: str
    ) -> str:
        return f"{system_prompt}\n" f"{prompt}\n\n" f"{json_instruction}"

    async def get_structured_response(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str = "",
    ) -> T:
        try:
            json_instruction = self.prompt_loader.get_prompt("ollama_json_instruction")
            full_prompt = self._build_full_prompt(
                system_prompt, prompt, json_instruction
            )

            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "format": "json",
                }

                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"Ollama API error {response.status}: {error_text}"
                        )

                    result = await response.json()
                    generated_text = result.get("response", "").strip()

                    if not generated_text:
                        raise RuntimeError("Empty response from Ollama")

                    try:
                        parsed_json = json.loads(generated_text)
                        return response_model.model_validate(parsed_json)
                    except json.JSONDecodeError as e:
                        raise RuntimeError(f"Invalid JSON response: {e}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to validate response: {e}")

        except Exception as e:
            raise RuntimeError(f"Failed to get structured response: {e}")


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
    ):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key is required.")

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        self.model = model

    def _clean_schema_for_gemini(self, schema: dict) -> dict:
        cleaned = {}

        for key, value in schema.items():
            # Skip unsupported fields
            if key in ["title", "$defs", "definitions", "default"]:
                continue
            elif key == "type":
                # Map type to type_
                cleaned["type_"] = value
            elif isinstance(value, dict):
                cleaned[key] = self._clean_schema_for_gemini(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    (
                        self._clean_schema_for_gemini(item)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]
            else:
                cleaned[key] = value

        return cleaned

    async def get_structured_response(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str = "",
    ) -> T:
        try:
            schema = response_model.model_json_schema()

            cleaned_schema = self._clean_schema_for_gemini(schema)

            generation_config = genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=cleaned_schema,
                temperature=0.0,
            )

            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

            response = await self.client.generate_content_async(
                full_prompt,
                generation_config=generation_config,
            )

            if not response.text:
                raise RuntimeError("Empty response from Gemini")

            try:
                parsed_json = json.loads(response.text)
                return response_model.model_validate(parsed_json)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON response: {e}")
            except Exception as e:
                raise RuntimeError(f"Failed to validate response: {e}")

        except Exception as e:
            raise RuntimeError(f"Failed to get structured response from Gemini: {e}")
