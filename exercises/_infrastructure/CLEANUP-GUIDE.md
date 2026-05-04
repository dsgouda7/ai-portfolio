# 🧹 Docker Cleanup Scripts — Complete Guide

## ✅ **What Was Created**

Two cleanup scripts to match the deployment workflow:
- **cleanup.ps1** (PowerShell/Windows)
- **cleanup.sh** (Bash/Linux/Mac)

Both provide **3 cleanup levels**:

---

## 🎯 **Usage Examples**

### **Level 1: Stop Containers Only (Safest)**
`powershell
cd exercises\_infrastructure
.\cleanup.ps1 -Exercise "01-ml/01-regression"
`
**Result:** Stops containers, keeps images

### **Level 2: Stop + Remove Exercise Images**
`powershell
.\cleanup.ps1 -Exercise "01-ml/01-regression" -All
`
**Result:** Stops containers + removes exercise image + asks about shared images (Prometheus/Grafana)

### **Level 3: Nuclear Option (Cleans Everything)**
`powershell
.\cleanup.ps1 -Everything
`
**Result:** Removes ALL Docker resources (containers, images, volumes, networks)
**⚠️ WARNING:** Affects ALL Docker projects on your machine!

### **Dry Run (See What Would Happen)**
`powershell
.\cleanup.ps1 -Exercise "01-ml/01-regression" -All -DryRun
`

---

## 📝 **Complete Workflow**

### **Deploy → Work → Cleanup Pattern:**

`powershell
# 1. Deploy
cd exercises\_infrastructure
.\deploy.ps1 -Exercise "01-ml/01-regression" -Dockerfile "Dockerfile.ml" -Port 5001 -Build

# 2. Work with your API
# http://localhost:5001

# 3. Stop when done
.\cleanup.ps1 -Exercise "01-ml/01-regression"

# 4. Free disk space (removes images)
.\cleanup.ps1 -Exercise "01-ml/01-regression" -All
`

---

## 🗑️ **What Gets Removed**

| Command | Containers | Images | Shared Images | Volumes | Networks |
|---------|-----------|--------|---------------|---------|----------|
| -Exercise NAME | ✅ Stop | ❌ Keep | ❌ Keep | ❌ Keep | ❌ Keep |
| -Exercise NAME -All | ✅ Remove | ✅ Remove | ⚠️ Ask | ❌ Keep | ❌ Keep |
| -Everything | ✅ Remove | ✅ Remove ALL | ✅ Remove ALL | ✅ Remove ALL | ✅ Remove ALL |

---

## 🔍 **Disk Space Check**

After cleanup, the script shows:
`
📊 Current Docker disk usage:
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          3         1         2.5GB     1.8GB (72%)
Containers      2         0         150MB     150MB (100%)
Local Volumes   1         0         500MB     500MB (100%)
`

---

## ⚠️ **Safety Features**

1. **Confirmation prompts** before nuclear cleanup
2. **Dry-run mode** to preview actions
3. **Separate handling** of shared images (won't break other exercises)
4. **Disk usage report** after cleanup

---

## 🐧 **Linux/Mac Users**

Same commands, different script:
`ash
cd exercises/_infrastructure
./cleanup.sh --exercise "01-ml/01-regression" --all
./cleanup.sh --everything  # Nuclear option
`

---

## 📋 **config.yaml Status**

**Found:** 14 config.yaml files across all exercises

**Used by:** ONLY \src/api.py\ (for API deployment)

**NOT used by:** \main.py\ (local training)

**Decision:** These files are **OPTIONAL**. You can delete them if you're only doing local training:

\\\powershell
# Delete all config.yaml files (if you don't need API deployment)
Get-ChildItem -Path "exercises" -Filter "config.yaml" -Recurse | Remove-Item
\\\

**Keep them if:**
- You plan to deploy APIs with Flask/FastAPI
- You're working on exercises 06-ai_infrastructure or 07-devops_fundamentals

**Delete them if:**
- You're only running \python main.py\ locally
- You want cleaner exercise directories

---

## ✅ **Updated Documentation**

Check these files for complete guides:
- **exercises/_infrastructure/README.md** (deployment + cleanup)
- **exercises/README.md** (top-level overview)
- **exercises/GETTING_STARTED.md** (quick start)

