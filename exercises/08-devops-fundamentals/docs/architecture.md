# System Architecture

## Overview
The ML DevOps Demo is a production-ready ML API built with DevOps best practices, demonstrating CI/CD pipelines, containerization, and infrastructure as code.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer (Ingress)                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Kubernetes Cluster (GKE/EKS)                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               ML API Pods (HPA: 3-10)               │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │    │
│  │  │  Flask   │  │  Flask   │  │  Flask   │          │    │
│  │  │   API    │  │   API    │  │   API    │          │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘          │    │
│  └───────┼─────────────┼─────────────┼────────────────┘    │
│          │             │             │                      │
│  ┌───────▼─────────────▼─────────────▼────────────────┐    │
│  │            ConfigMap & Secrets                      │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  Monitoring Stack                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Prometheus  │  │   Grafana    │  │    Logs      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

## Components

### 1. Application Layer
- **Flask API**: Lightweight REST API for ML predictions
- **ML Model**: Scikit-learn RandomForest model
- **Health Checks**: Liveness and readiness probes
- **Metrics**: Prometheus metrics exposition

### 2. Container Layer
- **Docker**: Multi-stage builds for production
- **Docker Compose**: Local development environment
- **Image Registry**: Container image storage

### 3. Orchestration Layer
- **Kubernetes**: Container orchestration
- **Kustomize**: Environment-specific configuration
- **HPA**: Horizontal Pod Autoscaling
- **Ingress**: Load balancing and SSL termination

### 4. Infrastructure Layer
- **Terraform**: Infrastructure as Code (IaC)
- **GKE/EKS**: Managed Kubernetes service
- **VPC**: Network isolation
- **Cloud SQL**: Managed database (optional)

### 5. Configuration Management
- **Ansible**: Automated deployment playbooks
- **ConfigMaps**: Application configuration
- **Secrets**: Sensitive data management

### 6. CI/CD Pipeline
- **GitHub Actions**: Automated testing and deployment
- **GitLab CI**: Alternative CI/CD platform
- **Jenkins**: Enterprise CI/CD option

### 7. Monitoring & Observability
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Logs**: Centralized logging

## Data Flow

1. **Client Request** → Ingress → Service → Pod
2. **Pod** reads ConfigMap/Secrets
3. **Flask API** receives request
4. **ML Model** makes prediction
5. **Metrics** sent to Prometheus
6. **Response** returned to client

## Scalability

- **Horizontal Scaling**: HPA scales pods based on CPU/memory
- **Vertical Scaling**: Resource limits adjustable per environment
- **Load Balancing**: Ingress distributes traffic across pods

## High Availability

- **Multi-Pod Deployment**: Minimum 3 replicas in production
- **Rolling Updates**: Zero-downtime deployments
- **Health Checks**: Automatic pod replacement on failure
- **Multi-Zone**: Pods distributed across availability zones

## Security

- **Non-Root Containers**: Security best practice
- **Secret Management**: Kubernetes Secrets for sensitive data
- **Network Policies**: Pod-to-pod communication control
- **RBAC**: Role-based access control
- **SSL/TLS**: Encrypted traffic via Ingress

## Environments

### Development
- 2 replicas
- Debug logging
- Rapid iteration
- Local Docker Compose

### Staging
- 3 replicas
- INFO logging
- Integration testing
- Pre-production validation

### Production
- 5+ replicas (HPA)
- WARNING logging
- Manual approval required
- High availability
