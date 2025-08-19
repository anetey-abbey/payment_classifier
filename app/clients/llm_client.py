import asyncio
import uuid
from typing import Dict, List

from app.clients.base_client import BaseLLMClient
from app.clients.gemini_client import GeminiClient as _GeminiClient
from app.clients.ollama_client import OllamaClient as _OllamaClient
from app.clients.openai_client import OpenAIClient as _OpenAIClient
from app.core.exceptions import LLMClientError, LLMParseError, LLMTimeoutError
from app.core.protocols import MetricsCollector, PromptProvider, StructuredLogger
from app.models.classification import (
    BaseLLMConfig,
    ClassificationResult,
    GeminiConfig,
    LLMProviderType,
    OllamaConfig,
)


class LLMClientFactory:
    def __init__(
        self,
        prompt_provider: PromptProvider,
        logger: StructuredLogger,
        metrics: MetricsCollector,
        model_configs: Dict[str, BaseLLMConfig],
    ):
        self.model_configs = model_configs
        self.creators = {
            LLMProviderType.OLLAMA: _OllamaClient,
            LLMProviderType.GEMINI: _GeminiClient,
            LLMProviderType.OPENAI: _OpenAIClient,
        }
        self._model_to_provider = self._build_model_provider_mapping()
        self.prompt_provider = prompt_provider
        self.logger = logger
        self.metrics = metrics

    def _build_model_provider_mapping(self) -> Dict[str, LLMProviderType]:
        """Build mapping from model names to their provider types."""
        mapping = {}
        for model_name in self.model_configs:
            if model_name.startswith("gemini-"):
                mapping[model_name] = LLMProviderType.GEMINI
            elif model_name.startswith("gpt-") or model_name in ["gpt-4o-mini"]:
                mapping[model_name] = LLMProviderType.OPENAI
            else:
                mapping[model_name] = LLMProviderType.OLLAMA
        return mapping

    async def create_client(self, model_name: str) -> BaseLLMClient:
        """Create an LLM client for the specified model.

        Args:
            model_name: The specific model to use (e.g., 'gemini-1.5-flash', 'qwen2.5:1.5b')

        Returns:
            BaseLLMClient: Configured client instance for the requested model

        Raises:
            LLMClientError: If the model is unknown or unsupported
        """
        config = self.model_configs.get(model_name)
        if not config:
            raise LLMClientError(f"Unknown model: {model_name}")

        provider_type = self._model_to_provider.get(model_name)
        if not provider_type:
            raise LLMClientError(f"Cannot determine provider for model: {model_name}")

        creator_func = self.creators.get(provider_type)
        if not creator_func:
            raise LLMClientError(f"Unknown provider type: {provider_type}")

        client = creator_func(
            config,
            self.prompt_provider,
            self.logger,
            self.metrics,
        )
        await client._setup()
        return client


class LLMClientManager:
    def __init__(
        self,
        factory: LLMClientFactory,
        logger: StructuredLogger,
    ):
        self.factory = factory
        self.logger = logger
        self._clients: Dict[str, BaseLLMClient] = {}
        self._lock = asyncio.Lock()

    async def get_client(self, model_name: str) -> BaseLLMClient:
        async with self._lock:
            if model_name not in self._clients:
                self.logger.info("Creating new LLM client", model=model_name)
                self._clients[model_name] = await self.factory.create_client(model_name)
            return self._clients[model_name]

    async def classify(
        self,
        model_name: str,
        payment_text: str,
        categories: List[str],
        use_search: bool = False,
        **kwargs,
    ) -> ClassificationResult:
        client = await self.get_client(model_name)
        correlation_id = kwargs.get("correlation_id", str(uuid.uuid4()))
        return await client.classify(
            payment_text, categories, correlation_id, use_search
        )

    async def close_all(self):
        async with self._lock:
            for client in self._clients.values():
                await client._cleanup()
            self._clients.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()
        return None
