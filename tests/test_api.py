import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.classification import PaymentClassification

client = TestClient(app)


@pytest.fixture
def sample_categories():
    return [
        "groceries",
        "transport",
        "subscription",
        "healthcare",
        "utilities",
        "entertainment",
        "business_expense",
        "other",
    ]


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
            "/api/v1/classify",
            json={
                "payment_text": "Walmart grocery purchase $45.67",
                "categories": ["groceries", "transport", "entertainment"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "groceries"
        assert data["reasoning"] == "This appears to be a grocery store purchase"
        assert data["search_used"] == False


def test_classify_payment_invalid_request():
    response = client.post("/api/v1/classify", json={})
    assert response.status_code == 422


def test_classify_payment_missing_categories():
    response = client.post("/api/v1/classify", json={"payment_text": "Test payment"})
    assert response.status_code == 422


def test_classify_payment_empty_categories():
    response = client.post(
        "/api/v1/classify", json={"payment_text": "Test payment", "categories": []}
    )
    assert response.status_code == 422


def test_classify_payment_too_many_categories():
    many_categories = [f"category_{i}" for i in range(25)]
    response = client.post(
        "/api/v1/classify",
        json={"payment_text": "Test payment", "categories": many_categories},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_classify_payment_service_error():
    with patch(
        "app.services.classification_service.ClassificationService.classify_payment",
        new_callable=AsyncMock,
    ) as mock_classify:
        mock_classify.side_effect = Exception("LLM service unavailable")

        response = client.post(
            "/api/v1/classify",
            json={"payment_text": "Test payment", "categories": ["groceries", "other"]},
        )

        assert response.status_code == 500
        assert "LLM service unavailable" in response.json()["detail"]


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests require RUN_INTEGRATION_TESTS=true and running LLM service",
)
def test_classify_payment_integration(sample_categories):
    """
    Integration test that hits the real endpoint with real LLM.
    Requires LM Studio or similar LLM service running on localhost:1234
    """
    response = client.post(
        "/api/v1/classify",
        json={
            "payment_text": "Walmart Supercenter grocery shopping $67.84",
            "categories": sample_categories,
        },
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
        data["category"] in sample_categories
    ), f"LLM returned invalid category '{data['category']}'. Valid categories: {sample_categories}"

    print(f"Success Response: {data}")
