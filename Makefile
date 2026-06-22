.PHONY: test lint format coverage run docker-build clean help

PYTHON ?= .venv/Scripts/python.exe

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast: ## Run tests excluding slow/eval tests
	$(PYTHON) -m pytest tests/ -v --tb=short -m "not slow"

lint: ## Lint with Ruff (non-blocking)
	$(PYTHON) -m ruff check . --select=E,F,W --ignore=E501,E741,W503 || true

format: ## Format code with Ruff
	$(PYTHON) -m ruff format .

format-check: ## Check formatting without modifying
	$(PYTHON) -m ruff format --check --diff . || true

coverage: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --tb=short \
		--cov=. --cov-report=term-missing --cov-report=html \
		--cov-fail-under=75

coverage-xml: ## Run tests with XML coverage (for CI)
	$(PYTHON) -m pytest tests/ -v --tb=short \
		--cov=. --cov-report=term-missing --cov-report=xml \
		--cov-fail-under=75

run: ## Start the FastAPI server
	$(PYTHON) -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

docker-build: ## Build Docker image
	docker build -t medical-assistant:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env medical-assistant:latest

clean: ## Clean Python caches
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	@rm -f coverage.xml .coverage
	@echo "Cleaned."
