import pytest
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8000"

@pytest.fixture
def sample_request():
    return {
        "payment_text": "Walmart grocery purchase $45.67",
        "categories": ["test", "boodschappen", "transport", "entertainment"],
    }

def test_api_health():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200

@pytest.mark.parametrize("model_name,model_type,use_search", [
    ("qwen2.5:1.5b", "local", False),
    ("qwen2.5:1.5b", "local", True),
    ("gemini-1.5-flash", "cloud", False),
])
def test_classification_endpoints(sample_request, model_name, model_type, use_search):
    data = {
        **sample_request,
        "model_type": model_type,
        "model_name": model_name,
        "use_search": use_search,
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/classify", json=data)
    
    assert response.status_code == 200
    result = response.json()
    assert "category" in result
    assert "reasoning" in result
    assert isinstance(result["category"], str)
    assert len(result["category"]) > 0