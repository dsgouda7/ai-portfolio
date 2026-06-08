.PHONY: help docker-build docker-run docker-clean local-setup local-full clean

help:
	@echo "Databricks RAG Pipeline - Microservices Architecture"
	@echo "===================================================="
	@echo ""
	@echo "Docker Commands (Production):"
	@echo "  make docker-build        Build all Docker images"
	@echo "  make docker-run          Run full orchestrated pipeline"
	@echo "  make docker-ingest       Run phase 1 only (ingestion)"
	@echo "  make docker-vectorize    Run phase 1+2 (up to vectorization)"
	@echo "  make docker-serve        Run phase 3 only (assumes data exists)"
	@echo "  make docker-clean        Remove all Docker artifacts"
	@echo ""
	@echo "Local Commands (Development):"
	@echo "  make local-setup         Setup all virtual environments"
	@echo "  make local-ingest        Run phase 1 locally"
	@echo "  make local-vectorize     Run phase 2 locally"
	@echo "  make local-serve         Run phase 3 locally"
	@echo "  make local-full          Run complete pipeline locally"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean               Remove generated data only"
	@echo "  make clean-all           Remove data + virtual environments"

# Docker targets
docker-build:
	docker-compose build

docker-run:
	docker-compose up

docker-ingest:
	docker-compose up ingest

docker-vectorize:
	docker-compose up ingest vectorize

docker-serve:
	docker-compose up serve

docker-clean:
	docker-compose down --rmi all --volumes

# Local development targets
local-setup:
	@echo "Setting up Phase 1: Ingestion..."
	@cd phase1-ingest && bash setup.sh
	@echo ""
	@echo "Setting up Phase 2: Vectorization..."
	@cd phase2-vectorize && bash setup.sh
	@echo ""
	@echo "Setting up Phase 3: Serving..."
	@cd phase3-serve && bash setup.sh
	@echo ""
	@echo "All phases ready for local execution"

local-ingest:
	@cd phase1-ingest && . venv/bin/activate && python src/ingest.py

local-vectorize:
	@cd phase2-vectorize && . venv/bin/activate && python src/vectorize.py

local-serve:
	@cd phase3-serve && . venv/bin/activate && uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload

local-full:
	@echo "Running complete local pipeline..."
	@$(MAKE) local-ingest
	@$(MAKE) local-vectorize
	@$(MAKE) local-serve

# Cleanup targets
clean:
	@echo "Removing generated data..."
	rm -rf data/delta_lake/*
	rm -rf data/chroma_db/*
	rm -rf data/raw/*.csv data/raw/*.json data/raw/*.parquet
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Data cleaned"

clean-all: clean
	@echo "Removing virtual environments..."
	rm -rf phase1-ingest/venv
	rm -rf phase2-vectorize/venv
	rm -rf phase3-serve/venv
	@echo "Full cleanup complete"
