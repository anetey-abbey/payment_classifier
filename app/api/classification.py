import os
from typing import Dict, Tuple

from fastapi import APIRouter, HTTPException, Request

from app.clients.llm_client import GeminiClient, LLMClient, OllamaClient
from app.schemas.classification import (
    ClassificationRequest,
    ModelType,
    PaymentClassification,
)
from app.services.classification_service import ClassificationService

router = APIRouter(prefix="/api/v1", tags=["classification"])


def create_llm_client(model_type: ModelType, model_name: str) -> LLMClient:
    try:
        if model_type == ModelType.LOCAL:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaClient(base_url=base_url, model=model_name)
        elif model_type == ModelType.CLOUD:
            return GeminiClient(model=model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        raise ValueError(
            f"Failed to create LLM client for {model_type} with model {model_name}: {str(e)}"
        )


def get_cached_client(
    clients_cache: Dict[Tuple[str, str], LLMClient],
    model_type: ModelType,
    model_name: str,
) -> LLMClient:
    cache_key = (model_type.value, model_name)

    if cache_key not in clients_cache:
        clients_cache[cache_key] = create_llm_client(model_type, model_name)

    return clients_cache[cache_key]


@router.post("/classify", response_model=PaymentClassification)
async def classify_payment(
    classification_request: ClassificationRequest,
    request: Request,
) -> PaymentClassification:
    try:
        llm_client = get_cached_client(
            request.app.state.llm_clients,
            classification_request.model_type,
            classification_request.model_name,
        )
        service = ClassificationService(llm_client)
        result = await service.classify_payment(classification_request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
