.PHONY: help install install-dev dev test lint format clean build run docker-build docker-run docker-dev

help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  install-dev Install dependencies development"
	@echo "  dev         Run development server"
	@echo "  test        Run tests"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo "  build       Build application"
	@echo "  run         Run production server"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run  Run Docker container"
	@echo "  docker-dev  Run development with Docker Compose"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	black --check --line-length 88 .
	isort --check-only .
	flake8 .
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
	docker-compose -f docker-compose.dev.yml up --build