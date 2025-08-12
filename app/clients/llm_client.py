import asyncio
import uuid
from typing import Dict, List, Optional, Union

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
        **configs: BaseLLMConfig,
    ):
        self.configs = configs
        self.creators = {
            LLMProviderType.OLLAMA: _OllamaClient,
            LLMProviderType.GEMINI: _GeminiClient,
            LLMProviderType.OPENAI: _OpenAIClient,
        }
        self.prompt_provider = prompt_provider
        self.logger = logger
        self.metrics = metrics

    def _override_model_in_config(
        self,
        config: BaseLLMConfig,
        provider_type: LLMProviderType,
        model_name: str,
    ) -> BaseLLMConfig:
        """Override the model in a config instance based on provider type.

        Args:
            config: Base configuration to copy and modify
            provider_type: The provider type to determine how to override
            model_name: The model name to set

        Returns:
            BaseLLMConfig: A copy of the config with the model overridden
        """
        config_copy = config.model_copy()

        if provider_type == LLMProviderType.GEMINI:
            assert isinstance(config_copy, GeminiConfig)
            config_copy.model = model_name  # type: ignore
        elif provider_type == LLMProviderType.OLLAMA:
            assert isinstance(config_copy, OllamaConfig)
            config_copy.model = model_name

        return config_copy

    async def create_client(
        self,
        provider_type: Union[str, LLMProviderType],
        model_name: Optional[str] = None,
    ) -> BaseLLMClient:
        """Create an LLM client with dynamic model selection.

        This method creates a client instance using the base configuration for the
        provider type, but allows overriding the specific model if model_name is
        provided. This enables runtime model selection while maintaining efficient
        client caching based on provider+model combinations.

        Args:
            provider_type: The LLM provider (e.g., 'gemini', 'ollama', 'openai')
            model_name: Optional specific model to use (e.g., 'gemini-1.5-flash').
                       If None, uses the default model from the provider config.

        Returns:
            BaseLLMClient: Configured client instance for the requested provider/model

        Raises:
            LLMClientError: If the provider type is unknown or unsupported
        """
        if isinstance(provider_type, str):
            provider_type = LLMProviderType(provider_type.lower())

        config = self.configs.get(provider_type.name.lower())
        creator_func = self.creators.get(provider_type)
        if not config or not creator_func:
            raise LLMClientError(f"Unknown provider type: {provider_type}")

        # Override model in config if model_name is provided
        if model_name and provider_type in (
            LLMProviderType.GEMINI,
            LLMProviderType.OLLAMA,
        ):
            config = self._override_model_in_config(config, provider_type, model_name)

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

    async def get_client(
        self,
        provider_type: Union[str, LLMProviderType],
        model_name: Optional[str] = None,
    ) -> BaseLLMClient:
        provider_key = (
            provider_type.value
            if isinstance(provider_type, LLMProviderType)
            else provider_type.lower()
        )

        # Create unique key based on provider and model
        key = f"{provider_key}:{model_name}" if model_name else provider_key

        async with self._lock:
            if key not in self._clients:
                self.logger.info(
                    "Creating new LLM client", provider=provider_key, model=model_name
                )
                self._clients[key] = await self.factory.create_client(
                    provider_type, model_name
                )
            return self._clients[key]

    async def classify(
        self,
        provider_type: Union[str, LLMProviderType],
        payment_text: str,
        categories: List[str],
        use_search: bool = False,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> ClassificationResult:
        client = await self.get_client(provider_type, model_name)
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

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        await self.close_all()
