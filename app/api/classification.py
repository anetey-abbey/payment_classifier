import os

from fastapi import APIRouter, Depends, HTTPException

from app.clients.llm_client import LLMClient, LocalLMStudioClient
from app.schemas.classification import PaymentClassification, PaymentData
from app.services.classification_service import ClassificationService

router = APIRouter(prefix="/api/v1", tags=["classification"])


def get_llm_client() -> LLMClient:
    client_type = os.getenv("LLM_CLIENT_TYPE", "local").lower()

    if client_type == "local":
        return LocalLMStudioClient()
    else:
        raise ValueError(f"Unsupported LLM client type: {client_type}")


def get_classification_service(
    llm_client: LLMClient = Depends(get_llm_client),
) -> ClassificationService:
    return ClassificationService(llm_client)


@router.post("/classify", response_model=PaymentClassification)
async def classify_payment(
    request: PaymentData,
    service: ClassificationService = Depends(get_classification_service),
) -> PaymentClassification:
    try:
        result = await service.classify_payment(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
