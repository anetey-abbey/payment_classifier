import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.classification import PaymentClassification

client = TestClient(app)


@pytest.fixture
def valid_categories():
    config_path = Path(__file__).parent.parent / "config" / "categories.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["payment_categories"]


@pytest.mark.asyncio
async def test_classify_payment_success():
    mock_result = PaymentClassification(
        category="groceries",
        reasoning="This appears to be a grocery store purchase",
        search_used=False,
    )

    with patch(
        "app.services.classification_service.ClassificationService.classify_payment",
        new_callable=AsyncMock,
    ) as mock_classify:
        mock_classify.return_value = mock_result

        response = client.post(
            "/api/v1/classify", json={"payment_text": "Walmart grocery purchase $45.67"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "groceries"
        assert data["reasoning"] == "This appears to be a grocery store purchase"
        assert data["search_used"] == False


def test_classify_payment_invalid_request():
    response = client.post("/api/v1/classify", json={})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_classify_payment_service_error():
    with patch(
        "app.services.classification_service.ClassificationService.classify_payment",
        new_callable=AsyncMock,
    ) as mock_classify:
        mock_classify.side_effect = Exception("LLM service unavailable")

        response = client.post(
            "/api/v1/classify", json={"payment_text": "Test payment"}
        )

        assert response.status_code == 500
        assert "LLM service unavailable" in response.json()["detail"]


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests require RUN_INTEGRATION_TESTS=true and running LLM service",
)
def test_classify_payment_integration(valid_categories):
    """
    Integration test that hits the real endpoint with real LLM.
    Requires LM Studio or similar LLM service running on localhost:1234
    """
    response = client.post(
        "/api/v1/classify",
        json={"payment_text": "Walmart Supercenter grocery shopping $67.84"},
    )

    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")

    if response.status_code != 200:
        error_data = response.json()
        print(f"Error: {error_data}")

    assert response.status_code == 200
    data = response.json()

    assert "category" in data
    assert "reasoning" in data
    assert "search_used" in data

    assert isinstance(data["category"], str)
    assert len(data["category"]) > 0
    assert isinstance(data["reasoning"], str)
    assert len(data["reasoning"]) > 0
    assert isinstance(data["search_used"], bool)

    assert (
        data["category"] in valid_categories
    ), f"LLM returned invalid category '{data['category']}'. Valid categories: {valid_categories}"

    print(f"Success Response: {data}")


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests require RUN_INTEGRATION_TESTS=true and running LLM service",
)
def test_classify_payment_integration_multiple(valid_categories):
    """Test multiple different payment types"""
    test_cases = [
        "Amazon Prime membership $14.99",
        "Shell gas station $45.23",
        "Netflix subscription $15.99",
        "Target groceries and household items $89.45",
    ]

    for payment_text in test_cases:
        response = client.post("/api/v1/classify", json={"payment_text": payment_text})

        assert response.status_code == 200
        data = response.json()
        assert data["category"]
        assert data["reasoning"]
        assert (
            data["category"] in valid_categories
        ), f"LLM returned invalid category '{data['category']}' for '{payment_text}'. Valid categories: {valid_categories}"
        print(f"'{payment_text}' -> '{data['category']}': {data['reasoning']}")


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests require RUN_INTEGRATION_TESTS=true and running LLM service",
)
def test_classify_payment_llm_category_validation(valid_categories):
    """
    Dedicated test to validate that LLM responses always use valid categories.
    Tests various edge cases and payment types to ensure category compliance.
    """
    test_cases = [
        ("Walmart Supercenter groceries $67.84", "groceries"),
        ("Shell gas station fuel $45.23", "transport"),
        ("Netflix monthly subscription $15.99", "subscription"),
        ("Doctor visit copay $25.00", "healthcare"),
        ("Electric bill payment $89.45", "utilities"),
        ("Starbucks coffee $5.50", "entertainment"),
        ("Office supplies for work $34.99", "business_expense"),
        ("Unknown merchant charge $12.34", "other"),
    ]

    invalid_categories_found = []

    for payment_text, expected_category_type in test_cases:
        response = client.post("/api/v1/classify", json={"payment_text": payment_text})

        assert response.status_code == 200, f"Failed to classify: {payment_text}"
        data = response.json()

        category = data["category"]
        if category not in valid_categories:
            invalid_categories_found.append(
                {
                    "payment_text": payment_text,
                    "invalid_category": category,
                    "response": data,
                }
            )

        print(
            f"✓ '{payment_text}' -> '{category}' (valid: {category in valid_categories})"
        )

    if invalid_categories_found:
        error_msg = (
            f"LLM returned {len(invalid_categories_found)} invalid categories:\n"
        )
        for item in invalid_categories_found:
            error_msg += (
                f"  - '{item['payment_text']}' -> '{item['invalid_category']}'\n"
            )
        error_msg += f"Valid categories: {valid_categories}"
        assert False, error_msg

    print(
        f"All {len(test_cases)} test cases returned valid categories from: {valid_categories}"
    )
