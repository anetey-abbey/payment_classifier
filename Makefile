.PHONY: help install install-dev dev test lint format clean build run docker-build docker-run docker-dev ci

help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  install-dev Install dependencies development"
	@echo "  dev         Run development server (port 8000)"
	@echo "  test        Run tests"
	@echo "  test-full   Run tests with integrations"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo "  build       Build application"
	@echo "  run         Run production server"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run  Run Docker container"
	@echo "  docker-dev  Run development with Docker Compose (Ollama + API)"
	@echo "  docker-stop Stop Docker services"
	@echo "  docker-status Check API service status"
	@echo "  ollama-pull Pull Ollama model"
	@echo "  ci          Run continuous integration (format, clean, lint, test-full)"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --cov=app --cov-report=term-missing

test-full:
	@echo "Running all tests including integrations (requires LLM service running)..."
	RUN_INTEGRATION_TESTS=true pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	black --check --line-length 88 .
	isort --check-only .
	mypy app/

format:
	black --line-length 88 .
	isort .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache

build:
	docker build -t payment-classifier .

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t payment-classifier .

docker-run:
	docker run -p 8000:8000 payment-classifier

docker-dev:
	docker compose -f docker-compose.dev.yml up --build

docker-stop:
	docker compose -f docker-compose.dev.yml down

docker-status:
	@echo "Checking if payment classifier service is running..."
	@curl -f http://localhost:8000/ || echo "Service not ready yet"

ollama-pull:
	@echo "Pulling Ollama model..."
	docker exec payment_classifier-ollama-1 ollama pull qwen2.5:1.5b

ci: format clean lint test-full
	@echo "CI pipeline completed successfully"