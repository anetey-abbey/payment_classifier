import os
from unittest.mock import Mock

import pytest
from dotenv import load_dotenv

from app.main import get_app
from app.models.classification import ClassificationRequest, ModelType
from app.services.classification_service import ClassificationService

load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    os.environ.setdefault("LLM_CLIENT_TYPE", "ollama")
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:1.5b")
    os.environ.setdefault("RUN_INTEGRATION_TESTS", "true")


@pytest.fixture
def mock_llm_clients_cache():
    """Mock LLM clients cache for service layer testing"""
    return {}


@pytest.fixture
def classification_service(mock_llm_clients_cache):
    """Classification service instance for testing"""
    return ClassificationService(mock_llm_clients_cache)


@pytest.fixture
def sample_classification_request_local():
    """Sample classification request for local model"""
    return ClassificationRequest(
        payment_text="Coffee at Starbucks $5.50",
        categories=["food", "entertainment", "transport"],
        model_type=ModelType.LOCAL,
        model_name="qwen2.5:1.5b",
        use_search=False,
    )


@pytest.fixture
def sample_classification_request_cloud():
    """Sample classification request for cloud model"""
    return ClassificationRequest(
        payment_text="Grocery shopping at Walmart",
        categories=["groceries", "household", "other"],
        model_type=ModelType.CLOUD,
        model_name="gemini-2.5-flash",
        use_search=False,
    )


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "valid_models": {
            "local": ["qwen2.5:1.5b", "qwen2.5:3b"],
            "cloud": ["gemini-2.5-flash", "gemini-pro"],
        }
    }


@pytest.fixture
def app_instance():
    """FastAPI app instance for testing (without TestClient)"""
    return get_app()
