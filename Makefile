.PHONY: help install test lint format run-api run-dashboard run-mlflow clean

help:
	@echo "SensorDrift-MLOps Development Commands"
	@echo "========================================"
	@echo "  make install          - Install dependencies"
	@echo "  make test             - Run unit tests"
	@echo "  make test-cov         - Run tests with coverage"
	@echo "  make lint             - Run code quality checks"
	@echo "  make format           - Format code with black"
	@echo "  make run-api          - Start FastAPI server"
	@echo "  make run-dashboard    - Start Dash dashboard"
	@echo "  make run-mlflow       - Start MLflow UI"
	@echo "  make clean            - Clean cache & build files"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	python -m py_compile src/**/*.py

format:
	black src/ tests/ --line-length=100

run-api:
	python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	python dashboard/app.py

run-mlflow:
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root mlruns/artifacts

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -r {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info 2>/dev/null || true

.DEFAULT_GOAL := help
