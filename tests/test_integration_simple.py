from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import get_app


@pytest.fixture
def app():
    return get_app()


@pytest.fixture
def client(app):
    # Setup app state manually since lifespan events don't run in TestClient
    from unittest.mock import Mock

    from app.clients.llm_client import LLMClientFactory, LLMClientManager
    from app.services.classification_service import ClassificationService

    # Create mock dependencies
    mock_prompt_provider = Mock()
    mock_logger = Mock()
    mock_metrics = Mock()

    # Create mock factory and manager
    mock_factory = LLMClientFactory(mock_prompt_provider, mock_logger, mock_metrics)
    mock_manager = LLMClientManager(mock_factory, mock_logger)
    app.state.classification_service = ClassificationService(mock_manager)

    return TestClient(app)


def test_classify_endpoint_integration(client):
    """Integration test: Full HTTP request/response cycle with one API call"""

    with patch(
        "app.services.classification_service.ClassificationService.classify"
    ) as mock_classify:
        from app.models.classification import PaymentClassification

        mock_result = PaymentClassification(
            category="food",
            reasoning="Coffee purchase at a cafe",
        )
        mock_classify.return_value = mock_result

        response = client.post(
            "/api/v1/classify",
            json={
                "payment_text": "Coffee at Starbucks $4.50",
                "categories": ["food", "entertainment", "transport"],
                "model_type": "local",
                "model_name": "qwen2.5:1.5b",
                "use_search": False,
            },
        )
        assert response.status_code == 200

        response_data = response.json()
        assert response_data["category"] == "food"
        assert "reasoning" in response_data
        assert len(response_data["reasoning"]) > 0
        assert "search_used" in response_data

        # Verify the service was called with correct parameters
        mock_classify.assert_called_once()
        call_args = mock_classify.call_args[0][
            0
        ]  # First positional argument (ClassificationRequest)
        assert call_args.payment_text == "Coffee at Starbucks $4.50"
        assert call_args.categories == ["food", "entertainment", "transport"]
        assert call_args.model_type.value == "local"
