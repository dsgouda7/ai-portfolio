# Makefile for ML DevOps Demo

.PHONY: help install test lint format build run docker-build docker-run k8s-deploy terraform-apply clean

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := pytest
DOCKER_IMAGE := ml-devops-demo
DOCKER_TAG := latest
ENVIRONMENT := dev
NAMESPACE := ml-$(ENVIRONMENT)

help: ## Show this help message
	@echo "ML DevOps Demo - Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test: ## Run tests with coverage
	$(PYTEST) tests/ -v --cov=src --cov-report=term --cov-report=html

test-fast: ## Run tests without coverage
	$(PYTEST) tests/ -v

lint: ## Run linting (flake8)
	flake8 src tests --max-line-length=100 --exclude=venv

format: ## Format code with black
	black src tests

format-check: ## Check code formatting
	black --check src tests

ci: lint format-check test ## Run all CI checks

dev: ## Run development server
	$(PYTHON) -m src.api

docker-build: ## Build Docker image
	docker build -f docker/Dockerfile.prod -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-build-dev: ## Build development Docker image
	docker build -f docker/Dockerfile.dev -t $(DOCKER_IMAGE):dev .

docker-run: ## Run Docker container
	docker run -p 5000:5000 -e ENVIRONMENT=$(ENVIRONMENT) $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-compose-up: ## Start services with docker-compose (dev)
	docker-compose -f docker/docker-compose.dev.yml up -d

docker-compose-down: ## Stop services with docker-compose
	docker-compose -f docker/docker-compose.dev.yml down

docker-compose-logs: ## View docker-compose logs
	docker-compose -f docker/docker-compose.dev.yml logs -f

k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -k kubernetes/overlays/$(ENVIRONMENT)/

k8s-deploy-dev: ## Deploy to dev environment
	$(MAKE) k8s-deploy ENVIRONMENT=dev

k8s-deploy-staging: ## Deploy to staging environment
	$(MAKE) k8s-deploy ENVIRONMENT=staging

k8s-deploy-prod: ## Deploy to production environment
	$(MAKE) k8s-deploy ENVIRONMENT=prod

k8s-status: ## Check Kubernetes deployment status
	kubectl get pods -n $(NAMESPACE)
	kubectl get svc -n $(NAMESPACE)

k8s-logs: ## View Kubernetes logs
	kubectl logs -f deployment/ml-api -n $(NAMESPACE)

k8s-rollback: ## Rollback Kubernetes deployment
	kubectl rollout undo deployment/ml-api -n $(NAMESPACE)

k8s-delete: ## Delete Kubernetes resources
	kubectl delete -k kubernetes/overlays/$(ENVIRONMENT)/

terraform-init: ## Initialize Terraform
	cd terraform && terraform init

terraform-plan: ## Run Terraform plan
	cd terraform && terraform plan -var="project_id=$(PROJECT_ID)" -var="environment=$(ENVIRONMENT)"

terraform-apply: ## Apply Terraform configuration
	cd terraform && terraform apply -var="project_id=$(PROJECT_ID)" -var="environment=$(ENVIRONMENT)" -auto-approve

terraform-destroy: ## Destroy Terraform infrastructure
	cd terraform && terraform destroy -var="project_id=$(PROJECT_ID)" -var="environment=$(ENVIRONMENT)" -auto-approve

ansible-deploy: ## Deploy with Ansible
	ansible-playbook -i ansible/inventory/$(ENVIRONMENT).ini ansible/playbooks/deploy.yml

ansible-rollback: ## Rollback with Ansible
	ansible-playbook -i ansible/inventory/$(ENVIRONMENT).ini ansible/playbooks/rollback.yml

health-check: ## Run health checks
	./scripts/health-check.sh http://localhost:5000

setup: ## Setup development environment
	./scripts/setup.sh

clean: ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

clean-docker: ## Clean Docker images and containers
	docker-compose -f docker/docker-compose.dev.yml down -v
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) || true
	docker system prune -f

all: install lint format-check test docker-build ## Run all build steps
