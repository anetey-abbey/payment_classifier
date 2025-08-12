import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.clients.base_client import DefaultMetricsCollector, DefaultStructuredLogger
from app.clients.llm_client import LLMClientFactory, LLMClientManager
from app.core.logging import log_shutdown, log_startup, setup_logging
from app.core.prompt_loader import PromptLoader
from app.models.classification import GeminiConfig, OllamaConfig, OpenAIConfig
from app.services.classification_service import ClassificationService


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log_startup()

    prompt_provider = PromptLoader()
    logger = DefaultStructuredLogger()
    metrics = DefaultMetricsCollector()

    configs = {
        "ollama": OllamaConfig(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_search_engine_id=os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
        ),
        "gemini": GeminiConfig(api_key=os.getenv("GOOGLE_API_KEY")),
        "openai": OpenAIConfig(),
    }

    factory = LLMClientFactory(
        prompt_provider,
        logger,
        metrics,
        **configs,
    )
    client_manager = LLMClientManager(factory, logger)

    app.state.classification_service = ClassificationService(client_manager)

    yield

    await client_manager.close_all()
    log_shutdown()


def get_lifespan_handler():
    return lifespan
