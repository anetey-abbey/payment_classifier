import os

from fastapi import APIRouter, Depends, HTTPException

from app.clients.llm_client import LLMClient, LocalLMStudioClient, OllamaClient
from app.schemas.classification import ClassificationRequest, PaymentClassification
from app.services.classification_service import ClassificationService

router = APIRouter(prefix="/api/v1", tags=["classification"])


def get_llm_client() -> LLMClient:
    client_type = os.getenv("LLM_CLIENT_TYPE", "local").lower()

    if client_type == "local":
        base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        model = os.getenv("LM_STUDIO_MODEL", "qwen3-8b-instruct")
        return LocalLMStudioClient(base_url=base_url, api_key=api_key, model=model)
    elif client_type == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        return OllamaClient(base_url=base_url, model=model)
    else:
        raise ValueError(f"Unsupported LLM client type: {client_type}")


def get_classification_service(
    llm_client: LLMClient = Depends(get_llm_client),
) -> ClassificationService:
    return ClassificationService(llm_client)


@router.post("/classify", response_model=PaymentClassification)
async def classify_payment(
    request: ClassificationRequest,
    service: ClassificationService = Depends(get_classification_service),
) -> PaymentClassification:
    try:
        result = await service.classify_payment(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
