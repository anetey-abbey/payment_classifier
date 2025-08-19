.PHONY: help install-dev dev test-full lint format clean docker-dev docker-stop docker-status ci

help:
	@echo "Available commands:"
	@echo "  install-dev Install dependencies development"
	@echo "  dev         Run development server (port 8000)"
	@echo "  test-full   Run tests with integrations"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo "  docker-dev  Run development with Docker Compose (Ollama + API)"
	@echo "  docker-stop Stop Docker services"
	@echo "  docker-status Check API service status"
	@echo "  ci          Run continuous integration (format, clean, lint, test-full)"

install-dev:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test-full:
	@echo "Running minimal integration tests (requires LLM service running)..."
	pytest tests/test_integration.py -v

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

docker-dev:
	docker compose -f docker-compose.dev.yml up --build

docker-stop:
	docker compose -f docker-compose.dev.yml down

docker-status:
	@echo "Checking if payment classifier service is running..."
	@curl -f http://localhost:8000/ || echo "Service not ready yet"

ci: format clean lint test-full
	@echo "CI pipeline completed successfully"