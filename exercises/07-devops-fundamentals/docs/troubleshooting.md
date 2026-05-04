# Troubleshooting Guide

## Common Issues and Solutions

### 1. Pod Not Starting

#### Symptoms
- Pod status: `CrashLoopBackOff`, `Error`, or `Pending`
- Application not accessible

#### Diagnosis
```bash
# Check pod status
kubectl get pods -n ml-dev

# Describe pod for events
kubectl describe pod <pod-name> -n ml-dev

# Check logs
kubectl logs <pod-name> -n ml-dev

# Check previous logs if pod crashed
kubectl logs <pod-name> -n ml-dev --previous
```

#### Common Causes and Solutions

**ImagePullBackOff**
```bash
# Check if image exists and is accessible
docker pull ml-devops-demo:latest

# Verify image name in deployment
kubectl get deployment ml-api -n ml-dev -o yaml | grep image
```

**Insufficient Resources**
```bash
# Check node resources
kubectl top nodes

# Increase resource limits
kubectl edit deployment ml-api -n ml-dev
```

**Configuration Error**
```bash
# Check ConfigMap
kubectl get configmap ml-api-config -n ml-dev -o yaml

# Check Secrets
kubectl get secret ml-api-secret -n ml-dev -o yaml
```

### 2. Service Not Accessible

#### Symptoms
- Cannot reach service endpoint
- Connection timeout

#### Diagnosis
```bash
# Check service
kubectl get svc ml-api -n ml-dev

# Check endpoints
kubectl get endpoints ml-api -n ml-dev

# Test from within cluster
kubectl run test-pod --rm -i --restart=Never --image=curlimages/curl -- \
  curl http://ml-api.ml-dev.svc.cluster.local/health
```

#### Solutions

**No Endpoints**
```bash
# Check if pods are running and ready
kubectl get pods -n ml-dev -l app=ml-api

# Check pod labels match service selector
kubectl get pods -n ml-dev --show-labels
kubectl get svc ml-api -n ml-dev -o yaml | grep selector
```

**Port Mismatch**
```bash
# Verify container port
kubectl get deployment ml-api -n ml-dev -o yaml | grep -A 2 ports

# Verify service port
kubectl get svc ml-api -n ml-dev -o yaml | grep -A 5 ports
```

### 3. High Memory Usage

#### Symptoms
- Pods being OOMKilled
- Memory usage > 90%

#### Diagnosis
```bash
# Check memory usage
kubectl top pods -n ml-dev

# Check memory limits
kubectl describe pod <pod-name> -n ml-dev | grep -A 5 Limits
```

#### Solutions

**Increase Memory Limits**
```yaml
# Edit deployment
resources:
  limits:
    memory: "2Gi"
  requests:
    memory: "1Gi"
```

**Memory Leak Investigation**
```bash
# Monitor memory over time
watch kubectl top pods -n ml-dev

# Check application logs for memory issues
kubectl logs <pod-name> -n ml-dev | grep -i memory
```

### 4. Deployment Fails

#### Symptoms
- Rolling update stuck
- Old pods not terminating

#### Diagnosis
```bash
# Check rollout status
kubectl rollout status deployment/ml-api -n ml-dev

# Check rollout history
kubectl rollout history deployment/ml-api -n ml-dev

# Describe deployment
kubectl describe deployment ml-api -n ml-dev
```

#### Solutions

**Rollback**
```bash
# Rollback to previous version
kubectl rollout undo deployment/ml-api -n ml-dev

# Rollback to specific revision
kubectl rollout undo deployment/ml-api -n ml-dev --to-revision=2
```

**Force Update**
```bash
# Restart deployment
kubectl rollout restart deployment/ml-api -n ml-dev

# Delete old pods manually
kubectl delete pod <pod-name> -n ml-dev
```

### 5. Health Check Failures

#### Symptoms
- Pods repeatedly restarting
- Health check returning 503

#### Diagnosis
```bash
# Check health endpoint
curl http://localhost:5000/health

# Check readiness endpoint
curl http://localhost:5000/ready

# Check probe configuration
kubectl describe pod <pod-name> -n ml-dev | grep -A 10 Liveness
```

#### Solutions

**Increase Initial Delay**
```yaml
livenessProbe:
  initialDelaySeconds: 60  # Give more time to start
  periodSeconds: 10
```

**Adjust Failure Threshold**
```yaml
readinessProbe:
  failureThreshold: 5  # Allow more failures
```

### 6. Performance Issues

#### Symptoms
- High latency
- Slow predictions

#### Diagnosis
```bash
# Check metrics
curl http://localhost:5000/metrics

# Check resource usage
kubectl top pods -n ml-dev

# Check HPA status
kubectl get hpa -n ml-dev
```

#### Solutions

**Scale Up**
```bash
# Manual scaling
kubectl scale deployment ml-api -n ml-dev --replicas=5

# Enable HPA
kubectl autoscale deployment ml-api -n ml-dev \
  --min=3 --max=10 --cpu-percent=70
```

**Optimize Code**
```python
# Profile code
import cProfile
cProfile.run('model.predict(X)')

# Use batch predictions
predictions = model.predict(X_batch)
```

### 7. CI/CD Pipeline Failures

#### Symptoms
- Tests failing
- Build errors
- Deployment errors

#### Diagnosis
```bash
# Check GitHub Actions logs
# Visit: https://github.com/your-repo/actions

# Check test output locally
./scripts/test.sh

# Check Docker build locally
./scripts/build.sh prod
```

#### Solutions

**Test Failures**
```bash
# Run tests with verbose output
pytest tests/ -vv

# Run specific test
pytest tests/test_api.py::test_health_endpoint -v
```

**Build Failures**
```bash
# Check Docker build logs
docker build -f docker/Dockerfile.prod -t ml-devops-demo:latest . --no-cache

# Validate Dockerfile syntax
docker build -f docker/Dockerfile.prod --check .
```

### 8. Terraform Issues

#### Symptoms
- Infrastructure provisioning fails
- State file issues

#### Diagnosis
```bash
# Check Terraform plan
terraform plan

# Validate configuration
terraform validate

# Check state
terraform show
```

#### Solutions

**State Lock Issues**
```bash
# Force unlock
terraform force-unlock <lock-id>
```

**Resource Conflicts**
```bash
# Import existing resources
terraform import google_container_cluster.primary <cluster-name>

# Refresh state
terraform refresh
```

### 9. Ansible Deployment Issues

#### Symptoms
- Playbook fails
- Connection errors

#### Diagnosis
```bash
# Run with verbose output
ansible-playbook -i ansible/inventory/dev.ini \
  ansible/playbooks/deploy.yml -vvv

# Check inventory
ansible-inventory -i ansible/inventory/dev.ini --list
```

#### Solutions

**Connection Issues**
```bash
# Test connectivity
ansible all -i ansible/inventory/dev.ini -m ping

# Check kubectl access
kubectl get nodes
```

### 10. Database Connection Issues

#### Symptoms
- Cannot connect to database
- Connection timeouts

#### Diagnosis
```bash
# Check database pod
kubectl get pods -l app=postgres -n ml-dev

# Test connection
kubectl exec -it <api-pod> -n ml-dev -- \
  psql -h postgres -U user -d mldb
```

#### Solutions

**Update Connection String**
```yaml
env:
  - name: DATABASE_URL
    value: "postgresql://user:pass@postgres:5432/mldb"
```

## Debugging Commands

```bash
# Get all resources
kubectl get all -n ml-dev

# Get events
kubectl get events -n ml-dev --sort-by='.lastTimestamp'

# Port forward for debugging
kubectl port-forward svc/ml-api 5000:80 -n ml-dev

# Execute command in pod
kubectl exec -it <pod-name> -n ml-dev -- /bin/bash

# Copy files from pod
kubectl cp ml-dev/<pod-name>:/app/logs/app.log ./app.log

# Check resource quotas
kubectl describe resourcequota -n ml-dev

# Check network policies
kubectl get networkpolicies -n ml-dev
```

## Getting Help

1. Check logs first: `kubectl logs <pod-name> -n ml-dev`
2. Check events: `kubectl get events -n ml-dev`
3. Review documentation: See `docs/` directory
4. Search issues: Check GitHub issues
5. Ask team: Post in #devops channel
