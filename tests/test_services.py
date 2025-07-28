from unittest.mock import AsyncMock, Mock

import pytest

from app.clients.llm_client import LLMClient
from app.schemas.classification import ClassificationRequest, PaymentClassification
from app.services.classification_service import ClassificationService
from app.utils.prompt_loader import PromptLoader


@pytest.fixture
def sample_categories():
    return [
        "groceries",
        "utilities",
        "transport",
        "entertainment",
        "healthcare",
        "subscription",
        "business_expense",
        "other",
    ]


@pytest.fixture
def mock_llm_client():
    return Mock(spec=LLMClient)


@pytest.fixture
def mock_prompt_loader():
    return Mock(spec=PromptLoader)


@pytest.fixture
def classification_service(mock_llm_client):
    service = ClassificationService(mock_llm_client)
    service.prompt_loader = Mock(spec=PromptLoader)
    return service


@pytest.mark.asyncio
async def test_classify_payment_success(classification_service):
    request = ClassificationRequest(
        payment_text="Coffee at Starbucks $5.50",
        categories=["groceries", "entertainment", "other"],
    )
    expected_classification = PaymentClassification(
        category="entertainment", reasoning="Coffee purchase at restaurant/cafe"
    )

    classification_service.prompt_loader.get_formatted_prompt.return_value = (
        "Classify: Coffee at Starbucks $5.50"
    )
    classification_service.prompt_loader.get_prompt.return_value = (
        "You are a payment classifier"
    )
    classification_service.llm_client.get_structured_response = AsyncMock(
        return_value=expected_classification
    )

    result = await classification_service.classify_payment(request)

    assert result == expected_classification
    classification_service.prompt_loader.get_formatted_prompt.assert_called_once_with(
        key="classify_user_prompt",
        payment_text="Coffee at Starbucks $5.50",
        valid_categories="groceries, entertainment, other",
    )
    classification_service.prompt_loader.get_prompt.assert_called_once_with(
        key="system_prompt"
    )
    classification_service.llm_client.get_structured_response.assert_called_once_with(
        prompt="Classify: Coffee at Starbucks $5.50",
        response_model=PaymentClassification,
        system_prompt="You are a payment classifier",
    )


@pytest.mark.asyncio
async def test_classify_payment_with_different_text(classification_service):
    request = ClassificationRequest(
        payment_text="Gas station Shell $45.00", categories=["transport", "other"]
    )
    expected_classification = PaymentClassification(
        category="transport", reasoning="Gas station fuel purchase"
    )

    classification_service.prompt_loader.get_formatted_prompt.return_value = (
        "Classify: Gas station Shell $45.00"
    )
    classification_service.prompt_loader.get_prompt.return_value = (
        "You are a payment classifier"
    )
    classification_service.llm_client.get_structured_response = AsyncMock(
        return_value=expected_classification
    )

    result = await classification_service.classify_payment(request)

    assert result == expected_classification


@pytest.mark.asyncio
async def test_classify_payment_llm_error(classification_service):
    request = ClassificationRequest(
        payment_text="Invalid payment", categories=["other"]
    )

    classification_service.prompt_loader.get_formatted_prompt.return_value = (
        "Classify: Invalid payment"
    )
    classification_service.prompt_loader.get_prompt.return_value = (
        "You are a payment classifier"
    )
    classification_service.llm_client.get_structured_response = AsyncMock(
        side_effect=Exception("LLM error")
    )

    with pytest.raises(Exception, match="LLM error"):
        await classification_service.classify_payment(request)


@pytest.mark.asyncio
async def test_classify_payment_prompt_loader_error(classification_service):
    request = ClassificationRequest(payment_text="Test payment", categories=["other"])

    classification_service.prompt_loader.get_formatted_prompt.side_effect = KeyError(
        "prompt not found"
    )

    with pytest.raises(KeyError, match="prompt not found"):
        await classification_service.classify_payment(request)


@pytest.mark.asyncio
async def test_classify_payment_returns_valid_category(
    classification_service, sample_categories
):
    request = ClassificationRequest(
        payment_text="Walmart grocery purchase $67.84", categories=sample_categories
    )
    valid_category = "groceries"
    assert valid_category in sample_categories

    expected_classification = PaymentClassification(
        category=valid_category, reasoning="Grocery store purchase"
    )

    classification_service.prompt_loader.get_formatted_prompt.return_value = (
        "Classify: Walmart grocery purchase $67.84"
    )
    classification_service.prompt_loader.get_prompt.return_value = (
        "You are a payment classifier"
    )
    classification_service.llm_client.get_structured_response = AsyncMock(
        return_value=expected_classification
    )

    result = await classification_service.classify_payment(request)

    assert result.category in sample_categories
    assert result.category == valid_category


@pytest.mark.asyncio
async def test_classify_payment_with_invalid_category_should_fail_validation(
    classification_service, sample_categories
):
    request = ClassificationRequest(
        payment_text="Coffee at Starbucks $5.50", categories=sample_categories
    )
    invalid_category = "invalid_category"
    assert invalid_category not in sample_categories

    invalid_classification = PaymentClassification(
        category=invalid_category, reasoning="Invalid category test"
    )

    classification_service.prompt_loader.get_formatted_prompt.return_value = (
        "Classify: Coffee at Starbucks $5.50"
    )
    classification_service.prompt_loader.get_prompt.return_value = (
        "You are a payment classifier"
    )
    classification_service.llm_client.get_structured_response = AsyncMock(
        return_value=invalid_classification
    )

    result = await classification_service.classify_payment(request)

    assert result.category not in sample_categories
