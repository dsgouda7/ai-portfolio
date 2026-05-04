# Ch.8 — Security & Secrets Management

> **The story.** In **2014**, a single GitHub commit exposed the private keys for **50,000 AWS accounts** when developers pushed code containing hardcoded credentials. The incident cost organizations millions in compromised resources and led to the creation of automated secret scanning in CI/CD pipelines. By 2018, Docker Secrets, Kubernetes Secrets, and cloud-native secret managers (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault) had become standard practice. Every production deployment you'll build depends on secrets — database passwords, API keys, TLS certificates — and the difference between secure and catastrophic is knowing **secrets are runtime configuration, never build artifacts**.
>
> **Where you are in the curriculum.** This is chapter eight of the DevOps Fundamentals track. You've containerized apps (Ch.1), orchestrated services (Ch.2-3), automated deployments (Ch.4), and monitored systems (Ch.5). Now you're securing production deployments. Every container you run needs credentials — database passwords, API tokens, cloud provider keys. One hardcoded secret in a Dockerfile means every image pushed to a registry is a security breach waiting to happen. This chapter teaches you to **separate secrets from code** and **rotate credentials without redeploying**.
>
> **Notation in this chapter.** `secret` — sensitive data requiring access control (passwords, keys, tokens); `.env` — environment variable file (local dev only, never committed); `Docker Secrets` — encrypted secret distribution in Swarm mode; `Kubernetes Secret` — base64-encoded secret mounted as volume or env var; `secret rotation` — replacing credentials without downtime; `RBAC` — role-based access control (who can access what); `pre-commit hook` — git hook that blocks commits containing secrets.

---

## 0 · The Challenge — Where We Are

> 💡 **The mission**: Deploy a **production Flask app with database access** satisfying 5 constraints:
> 1. **NO SECRETS IN GIT**: Database password never appears in version control
> 2. **NO SECRETS IN IMAGES**: Docker image can be public — no credentials leaked
> 3. **RUNTIME INJECTION**: Secrets loaded at container startup from secure store
> 4. **ROTATION READY**: Can change password without rebuilding images
> 5. **AUDIT TRAIL**: Know who accessed secrets and when

**What we know so far:**
- ✅ We've containerized a Flask app (Ch.1)
- ✅ We've deployed it with Docker Compose (Ch.2)
- ✅ We've pushed images to registries (Ch.4)
- ❌ **But the database password is hardcoded in the Dockerfile!** Anyone with image access has credentials

**What's blocking us:**
We need **secrets management** — a secure way to provide credentials to running containers without embedding them in code or images. Without proper secrets handling:
- Hardcoded passwords in Dockerfiles leak when images are pushed to public registries
- Environment variables in docker-compose.yml are visible in process lists and logs
- Developers copy-paste credentials into Slack or email for debugging
- Rotating a compromised password requires rebuilding and redeploying every image
- No audit trail of who accessed what secret when

**What this chapter unlocks:**
The **secrets management workflow** — secure credentials from creation to revocation:
- **Build time**: No secrets in Dockerfile, docker-compose.yml, or git
- **Runtime**: Secrets injected via environment variables, mounted files, or secret stores
- **Rotation**: Update credentials in the secret store, restart containers — no rebuild
- **Audit**: Track secret access, enforce least privilege with RBAC

✅ **This is the security foundation** — every production deployment requires proper secrets handling.

---

## Animation

![Secrets lifecycle](img/ch08-secrets-lifecycle.gif)

> **What you're seeing:** The full lifecycle of a production secret — **Create** (generate in secret store) → **Store** (encrypted at rest) → **Access** (injected into container at runtime) → **Rotate** (update without downtime) → **Revoke** (remove access immediately). Each frame shows the security properties at that stage. The animation demonstrates that secrets never touch git, images, or build-time configuration — they're always runtime dependencies fetched from secure stores. This is the mental model for every production credential.

---

## 1 · Secrets Are Runtime Configuration, Not Build-Time Artifacts

Secrets are **data your application needs at runtime** but **must never be part of the build**. When you write a Dockerfile and run `docker build`, the resulting image should be **publishable to a public registry** without leaking credentials. This means:

- **No `ENV DB_PASSWORD=secret123` in Dockerfile** — environment variables at build time are baked into image layers
- **No `COPY .env /app/.env` in Dockerfile** — copying secret files into the image makes them visible in `docker history`
- **No hardcoded strings in source code** — `conn_string = "postgres://user:password@db:5432"` is a security breach

**The core principle:**
Secrets are **passed to containers**, not **embedded in containers**. At runtime, the container receives credentials through one of these mechanisms:
- **Environment variables** — `docker run -e DB_PASSWORD=secret123` (visible in `docker inspect`, use only for local dev)
- **Mounted secret files** — Docker Secrets, Kubernetes Secrets (files appear in `/run/secrets/`)
- **Secret stores** — Azure Key Vault, AWS Secrets Manager, HashiCorp Vault (fetched at startup via SDK)

The image itself contains **zero secrets**. Change the database password? Just update the secret store and restart containers — no rebuild, no redeploy.

---

## 1.5 · The Practitioner Workflow — Your 4-Phase Security Audit

> ⚠️ **Two ways to read this chapter:**
> - **Theory-first (recommended for learning):** Read §0→§6 sequentially to understand the concepts, then use this workflow as your reference
> - **Workflow-first (practitioners with existing knowledge):** Use this diagram as a jump-to guide when securing real deployments
>
> **Note:** Section numbers don't follow phase order because the chapter teaches concepts pedagogically (theory before application). The workflow below shows how to APPLY those concepts.

**What you'll build by the end:** A SOC 2-compliant deployment where secrets never touch git, images, or build-time configuration — only runtime injection from secure stores, with automated scanning preventing accidental leaks and 90-day rotation schedules.

```
Phase 1: AUDIT              Phase 2: LOCAL DEV         Phase 3: PRODUCTION        Phase 4: ROTATION
────────────────────────────────────────────────────────────────────────────────────────────────────
Scan for hardcoded secrets: Use .env files locally:    Runtime secret injection:  Rotate + prevent leaks:

• Run gitleaks on repo      • .env file (gitignored)   • Docker Secrets (Swarm)   • Update Key Vault secret
• Check Dockerfiles for ENV • python-dotenv in code    • K8s Secrets (volumes)    • Restart containers
• Search code for patterns  • Never commit .env        • Azure Key Vault / AWS    • Pre-commit hook blocks
• Review docker-compose.yml • Local only — never prod  • Secrets Manager (SDK)      accidental commits

→ DECISION:                 → DECISION:                → DECISION:                → DECISION:
  Found secrets in code?      Local dev setup?           Production platform?       Rotation schedule?
  • Yes: Remove + rotate      • Single dev: .env         • Docker Swarm: Secrets    • SOC 2: 90 days
  • Dockerfiles: Strip ENV    • Team: docker-compose     • Kubernetes: K8s Secrets  • PCI-DSS: 90 days
  • History: git-filter-repo    dev secrets              • Cloud: Key Vault/AWS     • Internal: 180 days
```

**The workflow maps to these sections:**
- **Phase 1 (AUDIT)** → §2.1 Hardcoded Secret Detection
- **Phase 2 (LOCAL DEV)** → §2.2 .env Files for Development
- **Phase 3 (PRODUCTION)** → §3 Mental Model → §3.1 Runtime Secret Injection
- **Phase 4 (ROTATION)** → §4.1 Secret Rotation Lifecycle

> 💡 **How to use this workflow:** Run Phase 1 (audit) immediately on any codebase you inherit or before first production deployment. Complete Phase 2 (local dev) once per project. Implement Phase 3 (production) before first deploy. Schedule Phase 4 (rotation) as a recurring calendar task (every 90 days for compliance).

---

## 2 · Securing a Flask App with Database Credentials

You're a backend engineer at a fintech startup. Your Flask API connects to a PostgreSQL database to store transaction records. The app works perfectly in development — you've hardcoded `DB_PASSWORD=dev123` in the Dockerfile for convenience. Your manager now requires **SOC 2 compliance**: no secrets in version control, all credential access must be audited, passwords must rotate every 90 days.

Your task: **refactor the deployment to use Docker Secrets in production and Azure Key Vault in the cloud**. The application code should never contain credentials. The CI/CD pipeline should block any commit that contains secret-like strings. The database password should be rotatable without downtime.

**The running example: Flask + PostgreSQL with secrets in 4 steps**

| Step | What you do | Why it matters |
|------|-------------|----------------|
| **1. AUDIT** | Scan repo for hardcoded secrets with `gitleaks` | Find all leaks before they reach production |
| **2. LOCAL DEV** | `DB_PASSWORD=dev123` in `.env`, loaded by `python-dotenv` | Keeps secrets out of code, `.env` is gitignored |
| **3. PRODUCTION** | Docker Secrets (Swarm) or K8s Secrets (volumes) | Secrets never in environment variables or images |
| **4. ROTATION** | Pre-commit hook + Key Vault rotation schedule | Prevents accidental commits + limits breach window |

By step 4, you have **SOC 2-compliant secrets handling**: no credentials in git, runtime-only injection, audit logs via secret store, rotation without redeployment.

---

## 2.1 · **[Phase 1: AUDIT]** Hardcoded Secret Detection

Before you can secure secrets, you need to **find them**. Legacy codebases often have credentials scattered across Dockerfiles, config files, and source code. The first step is a comprehensive scan.

**Scanning tools:**
- **gitleaks** — scans git history for secret patterns (API keys, passwords, tokens)
- **detect-secrets** — Yelp's tool for finding secrets in code and preventing commits
- **truffleHog** — searches through git history for high-entropy strings

**Code snippet — Phase 1: Scan repository for hardcoded secrets:**

```bash
# Use gitleaks to scan entire repository history
# This finds secrets in current files AND past commits (even if deleted)
docker run --rm -v $(pwd):/repo zricethezav/gitleaks:latest detect \
  --source /repo \
  --report-path /repo/gitleaks-report.json \
  --verbose

# Output format:
# Finding:     AWS Access Key
# Secret:      AKIA****************EXAMPLE  
# File:        docker-compose.yml
# Line:        14
# Commit:      a3f8d92 (2023-04-15)
# Author:      dev@example.com

# DECISION LOGIC
if [ -s gitleaks-report.json ]; then
  echo "❌ SECRETS FOUND — Review gitleaks-report.json"
  echo "Action required:"
  echo "  1. Remove secrets from current files"
  echo "  2. Rotate all compromised credentials immediately"
  echo "  3. Use git-filter-repo to purge from history (if repo is private)"
  exit 1
else
  echo "✅ No secrets detected — Safe to proceed"
fi
```

> 💡 **Industry Standard:** `gitleaks` (open-source, actively maintained)
> ```bash
> # Install locally (macOS)
> brew install gitleaks
> 
> # Run scan
> gitleaks detect --source . --verbose
> ```
> **When to use:** Run before every production deployment, integrate into CI/CD pipeline to block merges with secrets.
> **Common alternatives:** `detect-secrets` (Yelp), `truffleHog` (deep history search), GitHub Secret Scanning (automatic for public repos)

**Common secret patterns to scan for:**

| Pattern | Regex | Risk Level |
|---------|-------|------------|
| AWS Access Key | `AKIA[0-9A-Z]{16}` | ❌ CRITICAL — full AWS account access |
| Private SSH key | `-----BEGIN (RSA|OPENSSH) PRIVATE KEY-----` | ❌ CRITICAL — server access |
| Generic password | `password\s*=\s*["'][^"']+["']` | ⚠️ HIGH — depends on context |
| API key | `api[_-]?key\s*=\s*["'][^"']+["']` | ⚠️ HIGH — service access |
| Database URL | `postgres://.*:.*@` | ⚠️ HIGH — DB credentials in connection string |

### 2.1.1 DECISION CHECKPOINT — Phase 1 Complete

**What you just saw:**
- Gitleaks scan output showing 3 hardcoded secrets (AWS key in `docker-compose.yml`, DB password in `config.py`, API key in `.env.example`)
- Secrets detected not just in current files but also in git history from 6 months ago
- Even deleted files still contain secrets in commit history — removal ≠ security

**What it means:**
- Any secret ever committed to git history is **permanently compromised** until rotated
- Public repos on GitHub: assume all historical secrets are already leaked (GitHub crawlers find them instantly)
- Private repos: lower immediate risk, but still rotate — ex-employees, compromised laptops, accidental public pushes
- `.env.example` files often contain real secrets (developers copy-paste for convenience) — treat as sensitive

**What to do next:**
→ **Immediate actions (do this now):**
  - Rotate all detected credentials in their respective services (AWS, database, API provider)
  - Remove secrets from current files — move to `.env` (gitignored) for local dev
  - Add `.env` to `.gitignore` if not already there

→ **Git history cleanup (if repo is private):**
  - Use `git-filter-repo` to purge secrets from all commits: `git filter-repo --path config.py --invert-paths`
  - Force-push cleaned history: `git push --force --all` (coordinate with team first!)
  - Not recommended for public repos (history already crawled by third parties)

→ **For our Flask example:** Move `DB_PASSWORD` to `.env` file (Phase 2 below) — never commit it again.

---

## 2.2 · **[Phase 2: LOCAL DEV]** .env Files for Development

Local development needs secrets (database passwords, API keys for testing), but those secrets should **never reach git**. The `.env` file pattern is the industry standard for local-only configuration.

**The .env workflow:**
1. Create `.env` file in project root (same directory as `app.py` or `manage.py`)
2. Add `.env` to `.gitignore` — this file is **never committed**
3. Provide `.env.example` template (committed) with placeholder values
4. Load `.env` in application code using `python-dotenv` library

**Code snippet — Phase 2: Load secrets from .env file:**

```python
# .env file (NOT committed — listed in .gitignore)
DATABASE_URL=postgresql://user:dev_password_123@localhost:5432/mydb
API_KEY=sk_test_abcdef1234567890
SECRET_KEY=local-dev-secret-do-not-use-in-prod

# .env.example file (committed — serves as template)
DATABASE_URL=postgresql://user:YOUR_PASSWORD_HERE@localhost:5432/mydb
API_KEY=sk_test_YOUR_API_KEY_HERE
SECRET_KEY=generate-a-random-secret

# app.py (Flask application)
from flask import Flask
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()  # Looks for .env in current directory

app = Flask(__name__)

# Access secrets via os.getenv() — never hardcoded
DATABASE_URL = os.getenv('DATABASE_URL')
API_KEY = os.getenv('API_KEY')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# DECISION LOGIC (fail fast if critical secrets missing)
if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL not set — check .env file exists")
if not API_KEY:
    raise RuntimeError("❌ API_KEY not set — check .env file exists")

print(f"✅ Connected to database: {DATABASE_URL.split('@')[1]}")  # Log host only, not password
print(f"✅ API key loaded: {API_KEY[:7]}...{API_KEY[-4:]}")      # Log first 7 + last 4 chars only

@app.route('/')
def index():
    return "Flask app running with secrets from .env"

if __name__ == '__main__':
    app.run(debug=True)
```

> 💡 **Industry Standard:** `python-dotenv` (Python), `dotenv` (Node.js), `godotenv` (Go)
> ```bash
> # Install
> pip install python-dotenv
> ```
> **When to use:** Always in local development. Never in production (use Docker Secrets / K8s Secrets / Key Vault instead).
> **Common alternatives:** `direnv` (shell-level), `envparse` (type-safe parsing), `pydantic-settings` (validation)

**File structure after Phase 2:**

```
project/
├── .env                 ← NOT in git (contains real secrets)
├── .env.example         ← Committed (template with placeholders)
├── .gitignore           ← Contains: .env
├── app.py               ← Loads from .env via load_dotenv()
├── docker-compose.yml   ← References ${ENV_VARS} (Phase 3)
└── requirements.txt     ← Contains: python-dotenv
```

### 2.2.1 DECISION CHECKPOINT — Phase 2 Complete

**What you just saw:**
- Application code loads secrets via `os.getenv('DATABASE_URL')` — no hardcoded strings
- `.env` file contains real development credentials — gitignored, never committed
- `.env.example` provides template for new developers — committed, no real secrets
- Logging sanitized: only log database host (not password), only log API key prefix/suffix

**What it means:**
- **Local development is now secure** — no secrets in code, git history clean
- New team members: clone repo → copy `.env.example` to `.env` → fill in their credentials → done
- `.env` file is **local-only** — each developer has their own copy with their own credentials
- Production deployments **cannot use .env** — that file doesn't exist in containers (it's gitignored)

**What to do next:**
→ **For team environments:**
  - Use `docker-compose` with `.env` file for local multi-container dev stacks
  - Example: Flask app + PostgreSQL + Redis all reading from same `.env`
  
→ **For production:**
  - Phase 3 (below) — switch to Docker Secrets (Swarm) or K8s Secrets (Kubernetes)
  - Azure/AWS: fetch secrets from Key Vault / Secrets Manager at container startup

→ **Common mistake to avoid:**
  - Don't copy `.env` file into Docker image (`COPY .env /app/` in Dockerfile)
  - Docker images are immutable — hardcoded .env means you've baked secrets into the image
  - Always pass secrets at runtime, never at build time

---

## 3 · Mental Model — Build-Time vs. Runtime vs. Secret Stores

> 💡 **The analogy that never fails:** **Build time** is like constructing a house — you don't install the safe combination in the walls. **Runtime** is like moving in — the combination is handed to you separately. **Secret stores** are like a bank vault — the combination is kept offsite, only accessible to authorized residents.

**Build time (Dockerfile):**
- Packages application code, dependencies, runtime (Python, Flask)
- **Should contain zero secrets** — the image can be pushed to Docker Hub publicly
- Environment variables set with `ENV` are **baked into the image**, visible in `docker history`
- If you ever need to rotate a secret, you'd have to rebuild and redeploy — unacceptable in production

**Runtime (Container startup):**
- Secrets injected via `-e` flag, Docker Secrets, or Kubernetes Secrets
- Container reads secrets from environment variables or mounted files (`/run/secrets/db_password`)
- Secrets are **ephemeral** — stop the container, the secrets disappear from memory
- Rotate a secret? Update the source, restart the container — no image rebuild

**Secret stores (Azure Key Vault, AWS Secrets Manager, HashiCorp Vault):**
- Centralized, encrypted storage for secrets
- Access control via RBAC (only authorized services can read specific secrets)
- Audit logs track every secret access (who, what, when)
- Secrets can be rotated in the store — all containers fetch the new value on restart or refresh

**The lifecycle:**
```
Developer writes code (no secrets)
    ↓
Dockerfile builds image (no secrets)
    ↓
Image pushed to registry (public or private, no secrets leaked)
    ↓
Container started with secret injection:
    • Local dev: .env file
    • Docker Compose: Docker Secrets
    • Kubernetes: K8s Secrets
    • Cloud: Azure Key Vault / AWS Secrets Manager
    ↓
Application reads secret at runtime
    ↓
Secret rotated in store (container restarts or refreshes)
      ✅ WHAT THIS PREVENTS:
      • Compromised credentials remain valid indefinitely (rotation limits exposure window to 90 days max)
      • Single leaked password grants permanent access (old password stops working after rotation)
      • Insider threats (ex-employee's cached credentials expire on rotation schedule)
      • Compliance violations (SOC 2, PCI-DSS require 90-day password rotation)
```

---

## 3.1 · **[Phase 3: PRODUCTION]** Runtime Secret Injection

Production deployments require **centralized, auditable, encrypted secret management**. The `.env` file pattern from Phase 2 doesn't scale — there's no audit trail, no encryption at rest, no RBAC. This section covers three production-grade approaches.

### Option 1: Docker Secrets (Docker Swarm)

Docker Swarm's native secret management encrypts secrets at rest and in transit, mounts them as read-only files in containers.

**Code snippet — Phase 3: Docker Compose secrets:**

```yaml
# docker-compose.yml (Swarm mode)
version: '3.8'

services:
  flask-app:
    image: myapp:latest
    secrets:
      - db_password
      - api_key
    environment:
      # Pass secret file paths to application
      DATABASE_PASSWORD_FILE: /run/secrets/db_password
      API_KEY_FILE: /run/secrets/api_key
    deploy:
      replicas: 3

secrets:
  db_password:
    file: ./secrets/db_password.txt  # Local file for dev
    # external: true                 # Or reference external secret in production
  api_key:
    file: ./secrets/api_key.txt
```

```python
# app.py — read secrets from mounted files
import os

def read_secret(secret_name):
    """Read secret from Docker Secrets mount point."""
    secret_path = f'/run/secrets/{secret_name}'
    try:
        with open(secret_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to environment variable (for local dev)
        return os.getenv(secret_name.upper())

# DECISION LOGIC (runtime environment detection)
if os.path.exists('/run/secrets/db_password'):
    # Running in Docker Swarm with secrets
    DB_PASSWORD = read_secret('db_password')
    API_KEY = read_secret('api_key')
    print("✅ Loaded secrets from Docker Secrets")
else:
    # Running locally or in environment without Docker Secrets
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    API_KEY = os.getenv('API_KEY')
    print("⚠️  Loaded secrets from environment variables (local dev only)")

if not DB_PASSWORD:
    raise RuntimeError("❌ DB_PASSWORD not available")
```

> 💡 **Industry Standard:** Docker Secrets (Swarm mode)
> ```bash
> # Create secret from file
> echo "my_secure_password" | docker secret create db_password -
> 
> # Create secret from stdin (more secure — doesn't write to disk)
> docker secret create api_key -
> [paste secret]
> [Ctrl+D]
> 
> # List secrets
> docker secret ls
> 
> # Deploy stack with secrets
> docker stack deploy -c docker-compose.yml myapp
> ```
> **When to use:** Docker Swarm production deployments. Encrypted at rest, RBAC via service constraints.
> **Limitations:** Swarm-specific (doesn't work with `docker-compose up` — requires Swarm mode)

### Option 2: Kubernetes Secrets

Kubernetes Secrets are base64-encoded (not encrypted by default!) but can be mounted as volumes or environment variables.

```yaml
# Create Kubernetes secret (imperative)
kubectl create secret generic db-credentials \
  --from-literal=password=my_secure_password \
  --from-literal=username=admin

# Or from file (declarative)
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
data:
  password: bXlfc2VjdXJlX3Bhc3N3b3Jk  # base64 encoded
  username: YWRtaW4=
```

```yaml
# deployment.yaml — mount secret as volume
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: flask
        image: myapp:latest
        volumeMounts:
        - name: db-secrets
          mountPath: /run/secrets
          readOnly: true
        env:
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
      volumes:
      - name: db-secrets
        secret:
          secretName: db-credentials
```

> 💡 **Industry Standard:** Kubernetes Secrets + External Secrets Operator
> ```bash
> # Install External Secrets Operator (syncs from cloud secret stores)
> helm install external-secrets external-secrets/external-secrets -n external-secrets-system
> 
> # Create SecretStore pointing to Azure Key Vault
> kubectl apply -f - <<EOF
> apiVersion: external-secrets.io/v1beta1
> kind: SecretStore
> metadata:
>   name: azure-keyvault
> spec:
>   provider:
>     azurekv:
>       vaultUrl: "https://my-vault.vault.azure.net"
>       authSecretRef:
>         clientId:
>           name: azure-creds
>           key: client-id
> EOF
> ```
> **When to use:** Kubernetes production clusters. Enables GitOps (secret definitions in git, values in cloud).
> **Critical:** Enable encryption at rest (`EncryptionConfiguration`) — base64 ≠ security!

### Option 3: Cloud Secret Managers (Azure Key Vault / AWS Secrets Manager)

Fetch secrets directly from cloud providers at application startup — most secure, full audit logs.

```python
# app.py — fetch secrets from Azure Key Vault at startup
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os

# DECISION LOGIC (environment-based secret source)
if os.getenv('AZURE_KEY_VAULT_URL'):
    # Production: fetch from Azure Key Vault
    vault_url = os.getenv('AZURE_KEY_VAULT_URL')
    credential = DefaultAzureCredential()  # Uses managed identity in Azure
    client = SecretClient(vault_url=vault_url, credential=credential)
    
    DB_PASSWORD = client.get_secret('db-password').value
    API_KEY = client.get_secret('api-key').value
    print(f"✅ Loaded secrets from Azure Key Vault: {vault_url}")
elif os.path.exists('/run/secrets/db_password'):
    # Docker Swarm: read from mounted files
    DB_PASSWORD = open('/run/secrets/db_password').read().strip()
    API_KEY = open('/run/secrets/api_key').read().strip()
    print("✅ Loaded secrets from Docker Secrets")
else:
    # Local dev: fallback to .env
    from dotenv import load_dotenv
    load_dotenv()
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    API_KEY = os.getenv('API_KEY')
    print("⚠️  Loaded secrets from .env (local dev only)")

if not DB_PASSWORD:
    raise RuntimeError("❌ No secret source available — check configuration")
```

> 💡 **Industry Standard:** Azure Key Vault (Azure), AWS Secrets Manager (AWS), HashiCorp Vault (self-hosted)
> ```bash
> # Azure Key Vault: Create secret
> az keyvault secret set \
>   --vault-name my-vault \
>   --name db-password \
>   --value "my_secure_password"
> 
> # Grant access to managed identity (for app running in Azure)
> az keyvault set-policy \
>   --name my-vault \
>   --object-id <app-managed-identity-id> \
>   --secret-permissions get list
> ```
> **When to use:** Always in cloud production (Azure, AWS, GCP). Full audit logs, automatic rotation, RBAC.
> **Common alternatives:** AWS Secrets Manager (AWS), GCP Secret Manager (Google Cloud), HashiCorp Vault (multi-cloud)

### 3.1.1 DECISION CHECKPOINT — Phase 3 Complete

**What you just saw:**
- Three production secret injection methods: Docker Secrets (Swarm), K8s Secrets (volumes), Key Vault (SDK)
- Application code adapts at runtime: checks `/run/secrets/` first (Docker), then Key Vault URL env var (cloud), then `.env` (local)
- No secrets in images — `docker history` shows zero credentials
- Secrets are ephemeral — container restart pulls fresh values from source

**What it means:**
- **Production deployments are now secure** — secrets injected at runtime, not baked into images
- **Secret rotation is decoupled from deployment** — update Key Vault → restart containers → new password active (no image rebuild, no CI/CD pipeline re-run)
- **Audit trail exists** — Key Vault logs every secret access (who, when, which secret)
- **RBAC enforced** — only authorized services (via managed identity or service account) can read specific secrets

**What to do next:**
→ **Choose secret source based on platform:**
  - Docker Swarm → Docker Secrets (`docker secret create`)
  - Kubernetes → K8s Secrets + External Secrets Operator (syncs from Key Vault)
  - Azure App Service / AKS → Azure Key Vault (use managed identity for auth)
  - AWS ECS / EKS → AWS Secrets Manager (use IAM roles for auth)

→ **Enable encryption at rest:**
  - Kubernetes: configure `EncryptionConfiguration` (base64 ≠ encrypted!)
  - Key Vault: encryption automatic, keys managed by Azure

→ **Set up RBAC:**
  - Principle of least privilege: each service gets only its required secrets
  - Example: web app reads `db-password`, but NOT `stripe-api-key` (that's for billing service only)

---

## 4 · What Can Go Wrong — Common Pitfalls

| Problem | Why it happens | How to fix |
|---------|----------------|------------|
| **Secrets in logs** | Application logs `DB_PASSWORD` during connection error | Sanitize logs: `logger.info(f"Connecting to {host}:****")` |
| **Base64 ≠ encryption** | Kubernetes secrets are base64-encoded, not encrypted at rest | Enable encryption at rest (EncryptionConfiguration in K8s) |
| **Secrets in environment variables** | `docker inspect` reveals all `-e` vars | Use Docker Secrets or mounted files instead |
| **Secrets in `.bash_history`** | `docker run -e PASSWORD=secret` is saved in shell history | Use `--env-file` or secret mounts |
| **Stale secrets** | Password rotated in Key Vault, but container still uses cached old value | Implement secret refresh (restart container or poll vault) |
| **Over-privileged access** | All containers can read all secrets | Use RBAC: each service gets only its required secrets |
| **No audit trail** | Can't determine who accessed a secret | Enable audit logs in Key Vault / Secrets Manager |

### Real-world example: The Docker Hub Leak

In 2019, a developer pushed a Docker image to Docker Hub with `ENV AWS_ACCESS_KEY_ID=AKIA...` in the Dockerfile. The image was public. Within hours, attackers used the key to spin up 100 EC2 instances for cryptocurrency mining. Cost: $20,000 in 3 days.

**The fix:** Never use `ENV` for secrets. Always inject at runtime.

---

## 4.1 · **[Phase 4: ROTATION]** Secret Lifecycle Management

Secrets don't last forever. Compliance frameworks (SOC 2, PCI-DSS) require **90-day rotation schedules**. Even without compliance, rotation limits the blast radius of compromised credentials — if a password leaks today, it's only valid for the next 60 days, not forever.

**Rotation workflow:**
1. Generate new secret (random password, new API key)
2. Update secret in Key Vault / Secrets Manager
3. Restart containers (or implement hot-reload if supported)
4. Revoke old secret after grace period (e.g., 7 days for gradual rollout)
5. Verify all services using new secret

**Code snippet — Phase 4: Automated secret rotation (Azure Key Vault):**

```bash
# rotate-secrets.sh — run this script every 90 days (or automate via cron/GitHub Actions)
#!/bin/bash
set -e

VAULT_NAME="my-production-vault"
SECRET_NAME="db-password"

# Step 1: Generate new password (random 32-char string)
NEW_PASSWORD=$(openssl rand -base64 32)

echo "🔄 Rotating secret: $SECRET_NAME"

# Step 2: Update secret in Azure Key Vault
az keyvault secret set \
  --vault-name "$VAULT_NAME" \
  --name "$SECRET_NAME" \
  --value "$NEW_PASSWORD" \
  --expires "$(date -u -d '+90 days' +%Y-%m-%dT%H:%M:%SZ)"  # Auto-expire in 90 days

echo "✅ Secret updated in Key Vault"

# Step 3: Update database password (example for PostgreSQL)
export PGPASSWORD="$NEW_PASSWORD"
psql -h db.example.com -U admin -d postgres -c \
  "ALTER USER admin WITH PASSWORD '$NEW_PASSWORD';"

echo "✅ Database password updated"

# Step 4: Restart containers to pick up new secret
kubectl rollout restart deployment/flask-app
kubectl rollout status deployment/flask-app --timeout=5m

echo "✅ Containers restarted — using new secret"

# Step 5: Wait 7 days before revoking old secret (grace period)
echo "⚠️  Old secret still valid for 7 days (grace period)"
echo "   Run 'revoke-old-secrets.sh' after verifying all services healthy"
```

**Preventing accidental secret commits (pre-commit hook):**

```yaml
# .pre-commit-config.yaml — place this file in repo root
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
        exclude: package-lock.json

# Install pre-commit hooks (run once per repo)
# pip install pre-commit
# pre-commit install

# Now every 'git commit' runs detect-secrets scan
# If secrets found → commit blocked
```

**Example blocked commit:**

```bash
$ git commit -m "Add database connection"

detect-secrets...........................................................Failed
- hook id: detect-secrets
- exit code: 1

Potential secrets detected in config.py:
  Line 14: Potential AWS Access Key (AKIA1234567890EXAMPLE)
  Line 18: Potential Password in URL (postgres://user:secret@db:5432)

Commit blocked. Remove secrets before committing.
```

> 💡 **Industry Standard:** `detect-secrets` (Yelp) + `pre-commit` framework
> ```bash
> # Install
> pip install detect-secrets pre-commit
> 
> # Generate baseline (initial scan, creates .secrets.baseline)
> detect-secrets scan > .secrets.baseline
> 
> # Install git hooks
> pre-commit install
> 
> # Test hook (runs on all files)
> pre-commit run --all-files
> ```
> **When to use:** Install in every repository before first commit. Blocks secrets at commit time (cheapest place to catch them).
> **Common alternatives:** `gitleaks` (pre-commit hook mode), `git-secrets` (AWS-specific), GitHub Secret Scanning (automatic for public repos)

### 4.1.1 DECISION CHECKPOINT — Phase 4 Complete

**What you just saw:**
- Automated rotation script: generate new password → update Key Vault → update database → restart containers → verify
- Pre-commit hook: `detect-secrets` scans every commit, blocks if secret patterns detected
- Grace period: old secret remains valid for 7 days (allows gradual rollout, catches missed services)
- Expiration dates: Key Vault secrets auto-expire after 90 days (forces rotation even if script not run)

**What it means:**
- **Secrets are time-limited** — even if compromised, they expire automatically (SOC 2 compliance achieved)
- **Rotation is automated** — no manual password updates, no forgotten credentials
- **Accidental leaks prevented** — pre-commit hook is the last line of defense before code reaches git
- **Audit trail complete** — Key Vault logs show: secret created (timestamp), accessed (by which service), rotated (by whom)

**What to do next:**
→ **Set rotation schedule:**
  - SOC 2 / PCI-DSS compliance: **90 days** (mandatory)
  - Internal policy: 180 days (if no compliance requirement)
  - High-risk secrets (prod DB, payment API): **30 days**
  - Add calendar reminder or automate via GitHub Actions cron

→ **Implement pre-commit hooks:**
  - Install `detect-secrets` + `pre-commit` in all repos (see code snippet above)
  - Configure CI/CD to run `detect-secrets` on every PR (belt-and-suspenders)
  - Add `.secrets.baseline` to git (tracks known false positives)

→ **Monitor secret access:**
  - Enable Azure Key Vault diagnostic logs → send to Log Analytics
  - Alert on: secret accessed by unauthorized service, secret accessed outside business hours, secret rotation overdue

→ **Common mistake to avoid:**
  - Don't rotate secrets without updating the database/API provider first
  - Order matters: (1) update destination system, (2) update Key Vault, (3) restart containers
  - Reverse order = downtime (containers fetch new secret before database knows about it)

---

## 5 · Progress Check — Can You Audit This Dockerfile for Security Issues?

You're reviewing a colleague's Dockerfile for a Node.js API:

```dockerfile
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
ENV DATABASE_URL=postgres://user:mypassword123@db:5432/prod
ENV API_KEY=sk_live_abcdef1234567890
CMD ["node", "server.js"]
```

**Task:** Identify **three** security issues and propose fixes.

<details>
<summary>Click to reveal answers</summary>

**Issues:**
1. **Hardcoded database password in `ENV DATABASE_URL`**  
   → **Fix:** Remove `ENV` line, pass `DATABASE_URL` at runtime via `-e` or Docker Secrets

2. **Hardcoded API key in `ENV API_KEY`**  
   → **Fix:** Store in secret manager (Key Vault, Secrets Manager), fetch at container startup

3. **Secrets are visible in image history**  
   → **Fix:** Run `docker history <image>` and you'll see both secrets in plaintext

**Secure version:**
```dockerfile
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
# No secrets in Dockerfile!
CMD ["node", "server.js"]
```

**Runtime injection:**
```bash
docker run -e DATABASE_URL=$DATABASE_URL -e API_KEY=$API_KEY my-api:latest
# Or use Docker Secrets / K8s Secrets
```

</details>

---

## 6 · Bridge to Future — Applying DevOps to AI/ML Deployments

Every concept in this chapter applies directly to **AI Infrastructure** deployments (Ch.9 onwards):

| DevOps Skill | AI/ML Application |
|--------------|-------------------|
| **Docker Secrets** | Secure API keys for OpenAI, Anthropic, Cohere in inference containers |
| **Kubernetes Secrets** | Store Azure OpenAI endpoint keys for distributed training jobs |
| **Pre-commit hooks** | Block commits containing `.env` with cloud provider credentials |
| **Secret rotation** | Rotate fine-tuning API keys without redeploying ML models |
| **RBAC** | Restrict which services can access production model endpoints |
| **Audit logs** | Track which team members accessed proprietary training data credentials |

**Next steps:**
- **Ch.9 (AI Infrastructure)**: Deploy LLM inference APIs with Azure Key Vault
- **Ch.10 (Model Serving)**: Secure ML model endpoints with API key rotation
- **Ch.11 (MLOps)**: Track experiment credentials in MLflow with secret backends

**The through-line:** Secrets management isn't just a DevOps checkbox — it's the foundation of **production AI security**. Every model deployment, every API call, every data pipeline requires credentials. This chapter taught you to handle them correctly.

---

## What's Next?

You've completed the DevOps Fundamentals track! You can:
1. **Containerize apps** (Docker)
2. **Orchestrate services** (Docker Compose, Kubernetes)
3. **Automate deployments** (CI/CD pipelines)
4. **Monitor systems** (Prometheus, Grafana)
5. **Secure credentials** (Secrets management)

**Continue to:** [AI Infrastructure Track](../../ai_infrastructure/README.md) — Apply these skills to deploying production AI systems.

**Practice:** Build a complete secure microservices stack — three services (Flask, FastAPI, Redis), secrets in Key Vault, CI/CD with GitHub Actions, monitoring with Prometheus.
