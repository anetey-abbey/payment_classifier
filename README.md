# Payment Transaction Classifier

**The Challenge:** Automatically categorizing financial transactions is difficult. Descriptions like "SQ *TAQUERIA EL BUEN" or "BOL.COM AMSTERDAM" are ambiguous and require real-world context that rule-based systems lack.


**Solution:**
Clean, fast API for classifying payment transactions using modern LLMs. Feed the API messy payment data, get back organized categories.

**Quick example:** `"SPOTIFY PREMIUM MONTHLY"` + `["Entertainment", "Utilities", "Food"]` → `"Entertainment"`

**Model support:**
- **Cloud**: OpenAI GPT5, Gemini 2.5 Flash/Pro.
- **Local**: Ollama models + optional Google Search.

## Key Features

**Multi-LLM Backend** - Unified interface for local Ollama and cloud APIs with model-specific configs.

**Web-Augmented Intelligence** - Connect Google Search API to local models for better context on unknown merchants.

**FastAPI Service** - REST endpoints with proper error handling, logging, request validation, and structured JSON.

**Containerized with Docker** - Docker Compose, including the Ollama service, for simple, reproducible deployments.

## Evaluation

Payment Transaction Classifier was tested on a private real world Dutch dataset of 1,000 payment transaction texts. The following model configurations were tested:

| Model            | Configuration | Accuracy | Recall | F1-Score |
| ---------------- | ------------- | -------- | ------ | -------- |
| Gemini 2.5 Flash | Cloud API     | 0.92     | 0.89   | 0.90     |
| qwen2.5:1.5      | Local only    | 0.73     | 0.68   | 0.70     |
| qwen2.5:1.5      | +Search       | 0.79     | 0.75   | 0.77     |

## Requirements
- Python 3.9+
- Docker Desktop (for local Ollama models)
- Optional: Google Search API key, vector database setup


## Quick Start

```bash
# Development with Docker Compose
make docker-dev

# Local development
pip install -e ".[dev]"
make dev

# Run continuous integration (format, clean, lint, test-full)
make ci
```

## API Usage

```bash
curl -X POST "http://localhost:8000/api/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "payment_text": "SPOTIFY PREMIUM MONTHLY",
    "categories": ["Entertainment", "Utilities", "Food", "Transportation"],
    "model_type": "local",
    "model_name": "qwen2.5:1.5b",
    "use_search": false
  }'
```

**Response:**
```json
{
  "category": "Entertainment",
  "reasoning": "Spotify is a music streaming service, clearly an entertainment expense",
  "search_used": False
}
```

## Project Structure

```
payment-classifier/
app
├── __init__.py
├── api
│   ├── __init__.py
│   ├── router.py
│   └── routes
│       ├── __init__.py
│       └── classification.py
├── clients
│   ├── __init__.py
│   ├── base_client.py
│   ├── gemini_client.py
│   ├── llm_client.py
│   ├── ollama_client.py
│   └── openai_client.py
├── core
│   ├── __init__.py
│   ├── config.py
│   ├── event_handlers.py
│   ├── exceptions.py
│   ├── logging.py
│   ├── prompt_loader.py
│   └── protocols.py
├── main.py
├── models
│   ├── __init__.py
│   └── classification.py
└── services
    ├── __init__.py
    ├── classification_service.py
    └── search_service.py
```
