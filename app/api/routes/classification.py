from fastapi import APIRouter, HTTPException, Request

from app.core.exceptions import LLMClientError, LLMParseError, LLMTimeoutError
from app.models.classification import ClassificationRequest, PaymentClassification
from app.services.classification_service import ClassificationService

router = APIRouter(prefix="/api/v1", tags=["classification"])


@router.post("/classify", response_model=PaymentClassification)
async def classify_payment(
    classification_request: ClassificationRequest,
    request: Request,
) -> PaymentClassification:

    service: ClassificationService = request.app.state.classification_service

    try:
        result = await service.classify(classification_request)
        return result

    except LLMTimeoutError as e:
        raise HTTPException(status_code=408, detail=f"Request timeout: {str(e)}")
    except LLMParseError as e:
        raise HTTPException(
            status_code=422, detail=f"Failed to parse LLM response: {str(e)}"
        )
    except LLMClientError as e:
        raise HTTPException(status_code=503, detail=f"LLM service error: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
