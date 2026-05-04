# Deployment Guide

## Prerequisites

- Docker installed
- Kubernetes cluster (GKE, EKS, or local minikube)
- kubectl configured
- Terraform (for infrastructure provisioning)
- Ansible (optional, for deployment automation)

## Local Development

### 1. Setup Environment
```bash
# Unix/macOS/Linux
chmod +x scripts/*.sh
./scripts/setup.sh
source venv/bin/activate

# Windows PowerShell
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

### 2. Run Locally
```bash
# Start API
python -m src.api

# Or use Docker Compose
docker-compose -f docker/docker-compose.dev.yml up
```

### 3. Test
```bash
./scripts/test.sh

# Manual test
curl http://localhost:5000/health
```

## Docker Deployment

### Build Image
```bash
# Development
./scripts/build.sh dev

# Production
./scripts/build.sh prod
```

### Run Container
```bash
docker run -p 5000:5000 \
  -e ENVIRONMENT=prod \
  -e LOG_LEVEL=INFO \
  ml-devops-demo:latest
```

## Kubernetes Deployment

### 1. Provision Infrastructure (Terraform)
```bash
cd terraform

# Initialize
terraform init

# Plan
terraform plan -var="project_id=my-project" -var="environment=dev"

# Apply
terraform apply -var="project_id=my-project" -var="environment=dev"
```

### 2. Deploy to Kubernetes
```bash
# Deploy to Dev
kubectl apply -k kubernetes/overlays/dev/

# Deploy to Staging
kubectl apply -k kubernetes/overlays/staging/

# Deploy to Production
kubectl apply -k kubernetes/overlays/prod/
```

### 3. Verify Deployment
```bash
# Check pods
kubectl get pods -n ml-dev

# Check service
kubectl get svc -n ml-dev

# Check logs
kubectl logs -f deployment/ml-api -n ml-dev

# Port forward for testing
kubectl port-forward svc/ml-api 5000:80 -n ml-dev
```

## Ansible Deployment

### 1. Configure Inventory
```bash
# Edit inventory files
vi ansible/inventory/dev.ini
```

### 2. Deploy
```bash
# Deploy to Dev
ansible-playbook -i ansible/inventory/dev.ini ansible/playbooks/deploy.yml

# Deploy to Production
ansible-playbook -i ansible/inventory/prod.ini ansible/playbooks/deploy.yml
```

### 3. Rollback
```bash
ansible-playbook -i ansible/inventory/prod.ini ansible/playbooks/rollback.yml
```

## CI/CD Pipeline

### GitHub Actions

1. **Push to `develop`** → CI runs → Deploy to Dev
2. **Push to `main`** → CI runs → Deploy to Staging
3. **Manual trigger** → Approval → Deploy to Production

### GitLab CI

1. Configure `.gitlab-ci.yml`
2. Set up GitLab Runner
3. Configure environment variables
4. Push to trigger pipeline

### Jenkins

1. Install Jenkins
2. Configure `Jenkinsfile`
3. Set up credentials
4. Create pipeline job

## Deployment Strategies

### Rolling Update (Default)
```bash
kubectl set image deployment/ml-api ml-api=ml-devops-demo:v2 -n ml-prod
kubectl rollout status deployment/ml-api -n ml-prod
```

### Blue-Green Deployment
```bash
# Deploy green
kubectl apply -f deployment-green.yaml

# Switch traffic
kubectl patch service ml-api -p '{"spec":{"selector":{"version":"green"}}}'

# Remove blue
kubectl delete -f deployment-blue.yaml
```

### Canary Deployment
```bash
# Deploy canary (10% traffic)
kubectl apply -f deployment-canary.yaml

# Monitor metrics
# If successful, scale canary to 100%
kubectl scale deployment ml-api-canary --replicas=5

# Remove old deployment
kubectl delete deployment ml-api-old
```

## Rollback

### Kubernetes Rollback
```bash
# View history
kubectl rollout history deployment/ml-api -n ml-prod

# Rollback to previous
kubectl rollout undo deployment/ml-api -n ml-prod

# Rollback to specific revision
kubectl rollout undo deployment/ml-api -n ml-prod --to-revision=3
```

### Script-based Rollback
```bash
./scripts/rollback.sh prod
```

## Health Checks

```bash
# Run health checks
./scripts/health-check.sh http://ml-api.example.com

# Manual checks
curl http://ml-api.example.com/health
curl http://ml-api.example.com/ready
curl http://ml-api.example.com/metrics
```

## Monitoring

### Access Prometheus
```bash
kubectl port-forward svc/prometheus 9090:9090
# Visit http://localhost:9090
```

### Access Grafana
```bash
kubectl port-forward svc/grafana 3000:3000
# Visit http://localhost:3000 (admin/admin)
```

## Troubleshooting

### Pod not starting
```bash
kubectl describe pod <pod-name> -n ml-dev
kubectl logs <pod-name> -n ml-dev
```

### Service not accessible
```bash
kubectl get endpoints ml-api -n ml-dev
kubectl get svc ml-api -n ml-dev
```

### Check resource usage
```bash
kubectl top pods -n ml-dev
kubectl top nodes
```

## Cleanup

### Delete Kubernetes Resources
```bash
kubectl delete -k kubernetes/overlays/dev/
```

### Destroy Infrastructure
```bash
cd terraform
terraform destroy -var="project_id=my-project" -var="environment=dev"
```
