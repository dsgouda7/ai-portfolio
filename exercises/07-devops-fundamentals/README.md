# Exercise 07: DevOps Fundamentals — Production ML System

> **Grand Challenge:** Build and deploy a production-grade ML API with CI/CD, containerization, infrastructure as code, and zero-downtime deployments.

**Scaffolding Level:** 🟡 Medium (focus on DevOps practices)

---

## Objective

Implement a complete DevOps pipeline for ML systems with:
- ✅ CI/CD pipelines (GitHub Actions, GitLab CI, Jenkins)
- ✅ Containerization (Docker multi-stage builds)
- ✅ Orchestration (Kubernetes with Kustomize)
- ✅ Infrastructure as Code (Terraform)
- ✅ Configuration Management (Ansible)
- ✅ Monitoring & Observability (Prometheus, Grafana)
- ✅ Automated testing and deployment
- ✅ Zero-downtime rolling updates and rollbacks

---

## What You'll Learn

### DevOps Fundamentals
- CI/CD pipeline design and implementation
- Container orchestration with Kubernetes
- Infrastructure provisioning with Terraform
- Configuration management with Ansible
- Deployment strategies (rolling, blue-green, canary)

### Production Best Practices
- Multi-stage Docker builds
- Health checks and readiness probes
- Horizontal Pod Autoscaling (HPA)
- Secret management
- Resource limits and quotas
- Logging and monitoring

### ML-Specific DevOps
- Model versioning and deployment
- A/B testing infrastructure
- Model monitoring and drift detection
- Feature store integration patterns
- MLOps pipeline automation

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│         CI/CD Pipeline (GitHub Actions)          │
│  Test → Build → Deploy Dev → Staging → Prod     │
└─────────────────┬────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────┐
│           Container Registry (Docker)            │
└─────────────────┬────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────┐
│     Kubernetes Cluster (GKE/EKS/AKS)            │
│  ┌──────────────────────────────────────┐       │
│  │   ML API Pods (Autoscaling 3-10)     │       │
│  │   + Prometheus Metrics               │       │
│  │   + Health/Readiness Probes          │       │
│  └──────────────────────────────────────┘       │
│  ┌──────────────────────────────────────┐       │
│  │   Monitoring (Prometheus + Grafana)  │       │
│  └──────────────────────────────────────┘       │
└──────────────────────────────────────────────────┘
```

---

## Project Structure

```
07-devops_fundamentals/
├── src/                          # Application source code
│   ├── api.py                    # Flask REST API
│   ├── model.py                  # ML model wrapper
│   ├── health.py                 # Health check handlers
│   └── utils.py                  # Utilities
├── tests/                        # Test suite
│   ├── test_model.py            # Model tests
│   ├── test_api.py              # API tests
│   └── test_integration.py      # Integration tests
├── docker/                       # Docker configuration
│   ├── Dockerfile.dev           # Development image
│   ├── Dockerfile.prod          # Production multi-stage build
│   ├── docker-compose.dev.yml   # Dev environment
│   └── docker-compose.prod.yml  # Production stack
├── kubernetes/                   # Kubernetes manifests
│   ├── base/                    # Base manifests
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── configmap.yaml
│   │   └── secret.yaml
│   └── overlays/                # Environment-specific
│       ├── dev/
│       ├── staging/
│       └── prod/
├── terraform/                    # Infrastructure as Code
│   ├── main.tf                  # GCP/AWS infrastructure
│   ├── variables.tf
│   └── outputs.tf
├── ansible/                      # Configuration management
│   ├── playbooks/
│   │   ├── deploy.yml
│   │   └── rollback.yml
│   └── inventory/
├── .github/workflows/            # CI/CD pipelines
│   ├── ci.yml                   # Test & build
│   ├── cd-dev.yml               # Deploy to dev
│   ├── cd-staging.yml           # Deploy to staging
│   └── cd-prod.yml              # Deploy to prod
├── scripts/                      # Automation scripts
│   ├── setup.sh                 # Environment setup
│   ├── test.sh                  # Run tests
│   ├── build.sh                 # Docker build
│   ├── deploy.sh                # Deployment
│   ├── rollback.sh              # Rollback
│   └── health-check.sh          # Health monitoring
├── docs/                         # Documentation
│   ├── architecture.md
│   ├── deployment.md
│   ├── monitoring.md
│   └── troubleshooting.md
├── config.yaml                   # Application config
├── Makefile                      # Build automation
└── requirements.txt              # Python dependencies
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- kubectl (Kubernetes CLI)
- Terraform (optional, for infrastructure)
- Ansible (optional, for automation)

### Local Development

**Unix/macOS/Linux:**
```bash
chmod +x scripts/*.sh
./scripts/setup.sh
source venv/bin/activate
```

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

**Using Makefile:**
```bash
make install    # Install dependencies
make dev        # Run development server
```

---

## Usage

### Running Locally

```bash
# Start API
python -m src.api

# Or with Make
make dev

# Test endpoints
curl http://localhost:5000/health
curl http://localhost:5000/ready
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'
```

### Running with Docker

```bash
# Build and run
make docker-build
make docker-run

# Or with docker-compose
make docker-compose-up
```

### Running Tests

```bash
# All tests with coverage
make test

# Fast tests (no coverage)
make test-fast

# Linting and formatting
make lint
make format-check
```

### Deploying to Kubernetes

```bash
# Deploy to dev
make k8s-deploy-dev

# Deploy to staging
make k8s-deploy-staging

# Deploy to production (requires approval)
make k8s-deploy-prod

# Check status
make k8s-status

# View logs
make k8s-logs

# Rollback if needed
make k8s-rollback
```

---

## CI/CD Pipelines

### GitHub Actions

Four automated workflows:

1. **CI Pipeline** (`.github/workflows/ci.yml`)
   - Runs on every push/PR
   - Linting, formatting, testing
   - Docker image build
   - Coverage reporting

2. **CD to Dev** (`.github/workflows/cd-dev.yml`)
   - Triggers on `develop` branch push
   - Deploys to dev environment
   - Automated smoke tests

3. **CD to Staging** (`.github/workflows/cd-staging.yml`)
   - Triggers on `main` branch push
   - Deploys to staging environment
   - Integration tests

4. **CD to Production** (`.github/workflows/cd-prod.yml`)
   - Manual trigger only
   - Requires approval from 2 team members
   - Blue-green deployment strategy
   - Automated rollback on failure

### Alternative CI/CD

- **GitLab CI**: `.gitlab-ci.yml`
- **Jenkins**: `Jenkinsfile`

---

## Deployment Strategies

### 1. Rolling Update (Default)
- Gradual replacement of old pods with new ones
- Zero downtime
- Automatic rollback on failure

```bash
kubectl set image deployment/ml-api ml-api=ml-devops-demo:v2
```

### 2. Blue-Green Deployment
- Deploy new version alongside old
- Switch traffic when ready
- Instant rollback capability

```bash
# Deploy green
kubectl apply -f deployment-green.yaml

# Switch traffic
kubectl patch service ml-api -p '{"spec":{"selector":{"version":"green"}}}'
```

### 3. Canary Deployment
- Deploy to small percentage of traffic
- Monitor metrics
- Gradually increase traffic

```bash
# Deploy canary (10% traffic)
kubectl apply -f deployment-canary.yaml
```

---

## Monitoring & Observability

### Prometheus Metrics

- `ml_predictions_total` - Total predictions
- `ml_prediction_duration_seconds` - Prediction latency
- `ml_errors_total` - Error count by type
- Standard system metrics (CPU, memory, etc.)

Access metrics:
```bash
curl http://localhost:5000/metrics
```

### Grafana Dashboards

```bash
# Access Grafana
kubectl port-forward svc/grafana 3000:3000
# Visit http://localhost:3000 (admin/admin)
```

### Health Checks

```bash
# Liveness probe
curl http://localhost:5000/health

# Readiness probe
curl http://localhost:5000/ready

# Automated health checks
make health-check
```

---

## Infrastructure as Code

### Terraform

Provision complete infrastructure on GCP/AWS:

```bash
# Initialize
make terraform-init

# Plan changes
make terraform-plan PROJECT_ID=my-project ENVIRONMENT=prod

# Apply infrastructure
make terraform-apply PROJECT_ID=my-project ENVIRONMENT=prod

# Destroy infrastructure
make terraform-destroy PROJECT_ID=my-project ENVIRONMENT=prod
```

Provisions:
- GKE/EKS cluster with autoscaling
- VPC and networking
- Cloud SQL database
- Load balancers
- IAM roles and policies

### Ansible

Automated deployment playbooks:

```bash
# Deploy
make ansible-deploy ENVIRONMENT=prod

# Rollback
make ansible-rollback ENVIRONMENT=prod
```

---

## Success Criteria

Your implementation should achieve:

- ✅ All tests passing (coverage > 80%)
- ✅ Linting and formatting checks pass
- ✅ Docker image builds successfully
- ✅ Kubernetes deployment succeeds
- ✅ Health checks return 200 OK
- ✅ Zero-downtime rolling updates
- ✅ Automated rollback on failure
- ✅ Prometheus metrics exposed
- ✅ CI/CD pipeline runs end-to-end
- ✅ <100ms p95 prediction latency

---

## Common Tasks

```bash
# Development
make install          # Install dependencies
make dev             # Run dev server
make test            # Run tests
make lint            # Check code quality
make format          # Format code

# Docker
make docker-build    # Build image
make docker-run      # Run container
make docker-compose-up  # Start all services

# Kubernetes
make k8s-deploy-dev     # Deploy to dev
make k8s-deploy-prod    # Deploy to prod
make k8s-status         # Check status
make k8s-logs           # View logs
make k8s-rollback       # Rollback deployment

# Infrastructure
make terraform-apply    # Provision infrastructure
make ansible-deploy     # Deploy with Ansible

# Cleanup
make clean              # Clean Python artifacts
make clean-docker       # Clean Docker resources
```

---

## Documentation

Comprehensive guides in `docs/`:

- [Architecture](docs/architecture.md) - System design and components
- [Deployment](docs/deployment.md) - Deployment strategies and procedures
- [Monitoring](docs/monitoring.md) - Metrics, logs, and alerts
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

---

## Learning Resources

### DevOps Concepts
- [12-Factor App](https://12factor.net/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

### CI/CD
- [GitHub Actions](https://docs.github.com/en/actions)
- [GitLab CI/CD](https://docs.gitlab.com/ee/ci/)
- [Jenkins Pipeline](https://www.jenkins.io/doc/book/pipeline/)

### Infrastructure as Code
- [Terraform Documentation](https://www.terraform.io/docs)
- [Ansible Documentation](https://docs.ansible.com/)

### Monitoring
- [Prometheus](https://prometheus.io/docs/introduction/overview/)
- [Grafana](https://grafana.com/docs/)

---

## Next Steps

After completing this exercise, explore:

1. **Advanced Deployment**
   - Implement canary deployments
   - Set up A/B testing infrastructure
   - Add feature flags

2. **Enhanced Monitoring**
   - Custom Grafana dashboards
   - Alert rules and notifications
   - Distributed tracing

3. **MLOps Pipeline**
   - Model versioning system
   - Automated model retraining
   - Model performance monitoring
   - Feature store integration

4. **Security Hardening**
   - Network policies
   - Pod security policies
   - Secret encryption
   - RBAC implementation

5. **Multi-Cloud**
   - Deploy to multiple cloud providers
   - Cross-region replication
   - Disaster recovery

---

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md) for common issues and solutions.

Quick diagnostics:
```bash
# Check pod status
kubectl get pods -n ml-dev

# Check logs
kubectl logs -f deployment/ml-api -n ml-dev

# Describe resources
kubectl describe pod <pod-name> -n ml-dev

# Run health checks
make health-check
```

---

## Resources

**Concept Review:**
- [notes/07-devops_fundamentals/](../../notes/07-devops_fundamentals/)
