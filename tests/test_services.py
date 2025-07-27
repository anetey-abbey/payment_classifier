from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from app.clients.llm_client import LLMClient
from app.schemas.classification import PaymentClassification, PaymentData
from app.services.classification_service import ClassificationService
from app.utils.prompt_loader import PromptLoader


@pytest.fixture
def valid_categories():
    config_path = Path(__file__).parent.parent / "config" / "categories.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["payment_categories"]


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
    payment_data = PaymentData(payment_text="Coffee at Starbucks $5.50")
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

    result = await classification_service.classify_payment(payment_data)

    assert result == expected_classification
    classification_service.prompt_loader.get_formatted_prompt.assert_called_once_with(
        key="classify_user_prompt",
        payment_text="Coffee at Starbucks $5.50",
        valid_categories="groceries, utilities, transport, entertainment, healthcare, subscription, business_expense, other",
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
    payment_data = PaymentData(payment_text="Gas station Shell $45.00")
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

    result = await classification_service.classify_payment(payment_data)

    assert result == expected_classification


@pytest.mark.asyncio
async def test_classify_payment_llm_error(classification_service):
    payment_data = PaymentData(payment_text="Invalid payment")

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
        await classification_service.classify_payment(payment_data)


@pytest.mark.asyncio
async def test_classify_payment_prompt_loader_error(classification_service):
    payment_data = PaymentData(payment_text="Test payment")

    classification_service.prompt_loader.get_formatted_prompt.side_effect = KeyError(
        "prompt not found"
    )

    with pytest.raises(KeyError, match="prompt not found"):
        await classification_service.classify_payment(payment_data)


@pytest.mark.asyncio
async def test_classify_payment_returns_valid_category(
    classification_service, valid_categories
):
    payment_data = PaymentData(payment_text="Walmart grocery purchase $67.84")
    valid_category = "groceries"
    assert valid_category in valid_categories

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

    result = await classification_service.classify_payment(payment_data)

    assert result.category in valid_categories
    assert result.category == valid_category


@pytest.mark.asyncio
async def test_classify_payment_with_invalid_category_should_fail_validation(
    classification_service, valid_categories
):
    payment_data = PaymentData(payment_text="Coffee at Starbucks $5.50")
    invalid_category = "invalid_category"
    assert invalid_category not in valid_categories

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

    result = await classification_service.classify_payment(payment_data)

    assert result.category not in valid_categories
