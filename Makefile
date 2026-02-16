.PHONY: help install setup clean test run export docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup       - Run setup script"
	@echo "  make clean       - Clean temporary files"
	@echo "  make test        - Run tests"
	@echo "  make run         - Run analysis"
	@echo "  make export      - Export results"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run in Docker"

install:
	pip install -r requirements.txt

setup:
	./setup.sh

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

test:
	pytest tests/ -v --cov=src

run:
	python scripts/run_analysis.py

export:
	python scripts/export_results.py

docker-build:
	docker-compose build

docker-run:
	docker-compose up