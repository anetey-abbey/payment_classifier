import os

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    os.environ.setdefault("LLM_CLIENT_TYPE", "ollama")
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:1.5b")
    os.environ.setdefault("RUN_INTEGRATION_TESTS", "true")


@pytest.fixture
def client():
    return TestClient(app)
