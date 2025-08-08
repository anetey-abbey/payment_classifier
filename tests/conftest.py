import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from app.main import app

load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    os.environ.setdefault("LLM_CLIENT_TYPE", "ollama")
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:1.5b")
    os.environ.setdefault("RUN_INTEGRATION_TESTS", "true")


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        app.state.llm_clients = {}
        yield test_client
