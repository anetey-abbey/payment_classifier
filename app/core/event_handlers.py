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

    model_configs = {
        "qwen2.5:1.5b": OllamaConfig(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model="qwen2.5:1.5b",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_search_engine_id=os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
        ),
        "gemini-1.5-flash": GeminiConfig(
            api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-1.5-flash"
        ),
        "gemini-1.5-pro": GeminiConfig(
            api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-1.5-pro"
        ),
        "gemini-2.5-flash": GeminiConfig(
            api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash"
        ),
        "gpt-4o-mini": OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini"
        ),
    }

    factory = LLMClientFactory(
        prompt_provider,
        logger,
        metrics,
        model_configs,
    )
    client_manager = LLMClientManager(factory, logger)

    app.state.classification_service = ClassificationService(client_manager)

    yield

    await client_manager.close_all()
    log_shutdown()


def get_lifespan_handler():
    return lifespan
