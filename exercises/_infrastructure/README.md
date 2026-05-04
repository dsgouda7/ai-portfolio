# Shared Infrastructure — 100% Centralized Deployment

> **Purpose:** ALL infrastructure lives here — zero Docker files in exercise folders  
> **Philosophy:** Exercise folders contain ONLY Python code + setup scripts  
> **Result:** Zero infrastructure duplication, plug-and-play deployment

---

## 🎯 **Overview**

This folder contains **100% of the infrastructure** for deploying exercises. Exercise folders have **no Docker-related files** — everything is centralized here.

**What's here:**
- `docker/` — 3 generic Dockerfiles (ML, Deep Learning, API)
- `deploy.ps1` / `deploy.sh` — Plug-and-play deployment scripts
- `monitoring/` — Shared Prometheus + Grafana configs

**What's NOT in exercise folders:**
- ❌ Dockerfile
- ❌ docker-compose.yml
- ❌ .dockerignore
- ❌ Makefile
- ❌ prometheus.yml

**Exercise folders contain ONLY:**
- ✅ Python code (`src/`, `tests/`)
- ✅ Setup scripts (`setup.ps1`, `setup.sh`)
- ✅ Dependencies (`requirements.txt`)

---

## 🚀 **Quick Start**

### **Deploy Any Exercise**

**PowerShell (Windows):**
```powershell
cd exercises/_infrastructure
.\deploy.ps1 -Exercise "01-ml/01-regression" -Dockerfile "Dockerfile.ml" -Port 5001 -Build
```

**Bash (Linux/Mac/WSL):**
```bash
cd exercises/_infrastructure
./deploy.sh --exercise "01-ml/01-regression" --dockerfile "Dockerfile.ml" --port 5001 --build
```

**Result:**
- Builds Docker image using shared Dockerfile
- Generates docker-compose.yml on-the-fly
- Starts API + Prometheus + Grafana
- API: http://localhost:5001
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### **View Logs**
```powershell
.\deploy.ps1 -Exercise "01-ml/01-regression" -Dockerfile "Dockerfile.ml" -Port 5001 -Logs
```

### **Stop Containers**
```powershell
.\deploy.ps1 -Exercise "01-ml/01-regression" -Dockerfile "Dockerfile.ml" -Port 5001 -Stop
```
docker-compose -f ../../_infrastructure/compose/base.yml up -d

# Access dashboards
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
```

---

### **3. Development Workflow**

```bash
# Local development (no Docker)
make setup           # Create venv, install dependencies
make test            # Run tests
python main.py       # Train models with immediate feedback

# Docker deployment (optional)
make docker-build    # Build container
make docker-up       # Start API + monitoring
make docker-logs     # View logs
make docker-down     # Stop containers
```

---

## 📊 **Port Allocation**

To avoid conflicts, each exercise uses a unique external port:

| Exercise | Port | Container Name |
|----------|------|----------------|
| 01-regression | 5001 | smartval-api |
| 02-classification | 5002 | classifier-api |
| 03-neural-networks | 5003 | nn-api |
| 04-recommender-systems | 5004 | recsys-api |
| 05-anomaly-detection | 5005 | anomaly-api |
| 06-reinforcement-learning | 5006 | rl-api |
| 07-unsupervised-learning | 5007 | unsupervised-api |
| 08-ensemble-methods | 5008 | ensemble-api |
| ... | ... | ... |

**Internal port:** Always 5000 (mapped via docker-compose)

---

## 🔧 **Customization**

### **Adding Exercise-Specific Dependencies**

**Option 1: requirements.txt (Recommended)**
```txt
# exercises/01-ml/01-regression/requirements.txt
xgboost>=1.7.0
catboost>=1.2.0
```

**Option 2: Custom Dockerfile (Rare)**
```dockerfile
# exercises/01-ml/01-regression/Dockerfile.custom
FROM ../../_infrastructure/docker/Dockerfile.ml

# Add exercise-specific dependencies
RUN pip install xgboost catboost
```

Then update Makefile:
```makefile
DOCKERFILE = Dockerfile.custom  # Override
```

---

### **Adding Exercise-Specific Prometheus Targets**

```yaml
# exercises/01-ml/01-regression/docker-compose.yml
services:
  smartval-api:
    # ... (existing config)
    labels:
      prometheus.io/scrape: "true"
      prometheus.io/port: "5000"
      prometheus.io/path: "/metrics"
```

Prometheus auto-discovers services with these labels.

---

## ⚙️ **Shared Makefile Targets**

| Target | Description |
|--------|-------------|
| `make help` | Show all available commands |
| `make setup` | Create venv and install dependencies |
| `make test` | Run pytest with coverage |
| `make lint` | Check code style (black, flake8) |
| `make format` | Auto-format code with black |
| `make clean` | Remove cache and generated files |
| `make serve` | Start Flask API locally |
| `make docker-build` | Build Docker image |
| `make docker-up` | Start containers (API + monitoring) |
| `make docker-down` | Stop containers |
| `make docker-logs` | View container logs |

**Custom targets:** Add exercise-specific targets after `include` statement.

---

## 🐛 **Troubleshooting**

### **"Cannot find Dockerfile"**
```bash
# Ensure relative path is correct
docker build -f ../../_infrastructure/docker/Dockerfile.ml .
```

### **"Port already in use"**
```bash
# Find process using port
netstat -ano | findstr :5001  # Windows
lsof -i :5001                 # Linux/Mac

# Stop conflicting container
docker ps
docker stop <container-id>
```

### **"Network ml-network not found"**
```bash
# Create network
docker network create ml-network

# Verify
docker network ls | grep ml-network
```

### **"Permission denied in container"**
```bash
# Ensure volumes have correct permissions
chmod -R 755 models logs  # Linux/Mac
```

---

## 📏 **Metrics**

**Before shared infrastructure:**
- Lines of infrastructure per exercise: ~300
- Total across 14 exercises: ~4,200 lines
- Maintenance burden: Change = 14 file edits

**After shared infrastructure:**
- Lines per exercise: ~15-20
- Total across 14 exercises: ~300 lines
- Maintenance burden: Change = 1 file edit

**Reduction:** 94% fewer infrastructure lines

---

## 🎯 **Philosophy**

> **"Infrastructure should just work — learners focus on ML code."**

- ✅ Standard infrastructure (Dockerfile, Makefile) is reusable
- ✅ Exercise-specific code (models.py, features.py) is unique
- ✅ Deployment is optional (Docker helps, but not required for learning)
- ✅ Immediate feedback prioritized (console output > container logs)

---

**Need help?** See per-exercise README or file an issue.
