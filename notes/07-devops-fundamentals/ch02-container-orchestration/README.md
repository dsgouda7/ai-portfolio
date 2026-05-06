# Ch.2 — Container Orchestration (Docker Compose)

> **Story:** Docker Compose emerged in 2014 (originally as Fig, acquired by Docker Inc.) to solve a universal frustration: managing multi-container applications with bash scripts. Before Compose, spinning up a web app + database + cache meant running 3+ `docker run` commands, remembering ports, networks, and environment variables. Sam Alba and the Docker team formalized the YAML-based declarative approach — define your entire stack once, start it with one command. Today, every engineer deploying microservices or multi-tier applications uses Compose daily — whether for local development or as a stepping stone to Kubernetes.
>
> **Where you are:** Ch.1 showed you how to containerize a single application with Docker — but production systems rarely run in isolation. You need a web server that talks to a database that connects to a cache layer. Managing these dependencies manually breaks down fast — port conflicts, startup ordering, network isolation. This chapter teaches you to orchestrate multiple containers as a single coordinated system, unlocking the ability to deploy real 3-tier architectures locally before scaling to cloud orchestrators like Kubernetes.
>
> **Notation:** `service` — a container definition in docker-compose.yml; `network` — isolated communication layer between services; `volume` — persistent storage shared across container restarts; `depends_on` — startup dependency declaration; `healthcheck` — automated readiness probe.

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Deploy a production-ready 3-tier web application (frontend → backend → database) that survives restarts, scales horizontally, and starts with one command.

**What we know so far:**
- ✅ Ch.1: We can containerize a single Python Flask app with Docker
- ❌ **But we can't coordinate multiple services** — manually running `docker run` for web + Postgres + Redis = 3 terminal windows, hardcoded IPs, brittle startup scripts

**What's blocking us:**

You've containerized a Flask API. Great. Now you need PostgreSQL for persistent data and Redis for session caching. The manual workflow:

```bash
docker run -d --name postgres -e POSTGRES_PASSWORD=secret postgres:16
docker run -d --name redis redis:7-alpine
docker run -d --name web --link postgres --link redis -p 5000:5000 my-flask-app
```

This breaks in three ways:
1. **Startup ordering**: The web container crashes if it starts before Postgres is ready to accept connections (race condition).
2. **Configuration drift**: Environment variables, ports, and volume mounts are scattered across 3 commands — one typo kills the stack.
3. **No persistence**: Stopping the Postgres container wipes your database unless you remember to manually create a named volume.

**What this chapter unlocks:**

Docker Compose — declarative multi-container orchestration. Define all services, networks, and volumes in one `docker-compose.yml` file. Start the entire stack with `docker compose up`, tear it down with `docker compose down`. Guaranteed startup ordering, automatic network isolation, persistent volumes. Production-grade 3-tier apps run locally with the same reliability as Kubernetes — but with zero cluster overhead.

---

## 1 · Core Idea — Declarative Service Orchestration

> ⚡ **When this breaks** — ProductionStack's Flask container starts before PostgreSQL finishes its initialization sequence; the app throws a `connection refused` error on the database and crashes, taking the payment service with it. Manually restarting containers in the right order during a 3 am outage is exactly how SLA violations happen. Docker Compose's `depends_on` + health check system eliminates startup race conditions by materializing services in the correct order, every time — no bash scripts, no sleep timers, no luck.

**The insight**: Don't script container startup — declare the *desired state* of your system.

A Docker Compose file is a blueprint:
- **Services** — each service is one container (or a scaled group of identical containers).
- **Networks** — isolated communication layers. Services on the same network can talk by name (DNS built-in).
- **Volumes** — persistent storage that survives `docker compose down`.

The Compose engine reads the YAML, creates networks and volumes, starts containers in dependency order, and monitors health. You get:
- **Idempotency**: Running `docker compose up` multiple times produces the same result.
- **Atomic teardown**: `docker compose down` removes all containers and networks (but preserves volumes unless you pass `--volumes`).
- **Environment parity**: The same `docker-compose.yml` runs on your laptop, CI server, and staging environment.

**Mental model**: Think of Compose as a *state manager*. You declare "I want 1 web container, 1 Postgres container, and 1 Redis container connected via a shared network," and Compose materializes that state. Stopping services returns the system to a clean slate (minus persistent volumes).

---

## 1.5 · The Practitioner Workflow — Your 4-Phase Orchestration

> ⚠️ **Two ways to read this chapter:**
> - **Theory-first (recommended for learning):** Read §0→§5 sequentially to understand concepts, then use this workflow as reference
> - **Workflow-first (practitioners with existing knowledge):** Use this diagram as a jump-to guide when deploying real stacks

**What you'll build by the end:** A production-ready 3-tier architecture (web + cache + database) that starts with one command, survives restarts, scales horizontally, and maintains persistent state. This is the complete docker-compose.yml from §2 that orchestrates service lifecycle, configuration, and observability.

```
Phase 1: DEFINE            Phase 2: CONFIGURE          Phase 3: ORCHESTRATE       Phase 4: OBSERVE
──────────────────────────────────────────────────────────────────────────────────────────────────
Declare service topology:  Inject runtime config:      Control lifecycle:         Monitor health:

• Service definitions      • Environment variables     • depends_on ordering      • Health checks
• Network topology         • .env files                • Restart policies         • Log aggregation
• Volume mounts            • Secrets management        • Resource limits          • Service discovery

→ DECISION:                → DECISION:                 → DECISION:                → DECISION:
  Service structure?         Config source?              Startup order?             Health criteria?
  • 3-tier: web/cache/db     • .env: local dev          • Strict: service_healthy   • DB: pg_isready
  • Networks: frontend/      • Vault: production        • Loose: service_started    • Web: HTTP 200
    backend isolation        • Never hardcode secrets   • Scale replicas with       • Logs: docker compose
  • Volumes: named for         in YAML                    --scale flag                logs -f <service>
    persistence
```

> 💡 **How to use this workflow:** Phase 1→2 define your stack architecture (run once per project). Phase 3→4 are iterative — run every deployment cycle. When a service fails to start (Phase 3), Phase 4 diagnostics (logs + health checks) identify the root cause before you modify Phase 1 or 2.

### The 4-Phase Decision Flow

This diagram shows the complete practitioner journey from single-container app to production-ready 3-tier orchestrated stack:

```mermaid
graph TD
    Start([docker-compose.yml]) --> B{Phase 1: DEFINE<br/>Service topology}
    B -->|No network isolation| C[Add custom bridge network]
    B -->|Data lost on restart| D[Add named volumes]
    B -->|Services declared| E{Phase 2: CONFIGURE<br/>Secrets and env vars}
    E -->|Secrets hardcoded in YAML| F[Move secrets to .env<br/>add .env to .gitignore]
    E -->|Config externalized| G{Phase 3: ORCHESTRATE<br/>docker compose up -d}
    G -->|Web crashes before db ready| H[Add depends_on with<br/>service_healthy condition]
    G -->|Container restart loop| I[docker compose logs -f web]
    G -->|Stack running| J{Phase 4: OBSERVE<br/>Health and logs}
    J -->|Service unhealthy| K[Fix health check command<br/>and retry interval]
    J -->|All services healthy| L[✅ ProductionStack ready]
    style L fill:#15803d,stroke:#15803d,color:#fff
    style B fill:#1e3a8a,stroke:#1e3a8a,color:#fff
    style E fill:#1e3a8a,stroke:#1e3a8a,color:#fff
    style G fill:#1e3a8a,stroke:#1e3a8a,color:#fff
    style H fill:#b45309,stroke:#b45309,color:#fff
    style I fill:#b91c1c,stroke:#b91c1c,color:#fff
```

---

## Animation

![Multi-container orchestration flow](img/ch02-compose-orchestration.gif)

> **What you're seeing:** Docker Compose startup sequence — Compose reads YAML → creates network → starts `cache` (no dependencies) → starts `db` (waits for health check) → starts `web` (waits for both). The animation demonstrates health-check-based dependency ordering preventing race conditions. This is the mental model for declarative orchestration: **declare desired state, Compose materializes it in correct order**.

---

## 2 · Running Example — Flask + PostgreSQL + Redis

**The system**: A Flask REST API (`/users` endpoint) that:
- Stores user records in **PostgreSQL** (persistent data)
- Caches frequently accessed queries in **Redis** (fast in-memory lookups)
- Exposes port 5000 externally

This is a canonical 3-tier architecture:
```
Client → Flask (web tier) → Redis (cache tier) + PostgreSQL (data tier)
```

### Step 1: Write `docker-compose.yml` — Service Composition

Create a new directory `flask-stack/` with:

```yaml
# docker-compose.yml
version: '3.9'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/appdb
      - REDIS_URL=redis://cache:6379
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_started
    networks:
      - app-network

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: appdb
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 5s
      timeout: 3s
      retries: 5
    networks:
      - app-network

  cache:
    image: redis:7-alpine
    networks:
      - app-network

networks:
  app-network:
    driver: bridge

volumes:
  postgres-data:
```

**Key details**:
- `depends_on.condition: service_healthy` ensures `web` waits for Postgres readiness (not just "container started").
- `networks: app-network` isolates services — external clients can't directly talk to `db` or `cache`.
- `volumes: postgres-data` persists database files across `docker compose down` cycles.

### Step 2: Define Service Dependencies

The `depends_on` block controls startup order:

```yaml
web:
  depends_on:
    db:
      condition: service_healthy  # Wait for health check to pass
    cache:
      condition: service_started  # Just wait for container to start
```

Without this, Flask would crash with "connection refused" errors on startup.

### Step 3: Configure Networks and Volumes

**Networks**: Services reference each other by name (`db`, `cache`) — Compose's internal DNS resolves these to container IPs automatically. The `app-network` is a private bridge network; only containers on the same network can communicate.

**Volumes**: The `postgres-data` named volume stores `/var/lib/postgresql/data` (Postgres's data directory). When you run `docker compose down`, the database persists. To wipe it: `docker compose down --volumes`.

> 💡 **Industry Standard:** Docker Compose YAML specification
> ```yaml
> # Canonical structure for production stacks
> services:
>   <service_name>:
>     image: <image>:<tag>  # Always pin versions (postgres:16-alpine, not postgres:latest)
>     depends_on:
>       <dependency>:
>         condition: service_healthy  # Wait for health checks, not just startup
>     networks:
>       - <network_name>
>     volumes:
>       - <named_volume>:<container_path>
> ```
> **When to use:** Always use declarative YAML instead of imperative `docker run` commands. YAML is version-controlled, reviewable, and reproducible.
> **Common alternatives:** Docker Swarm (deprecated), Kubernetes (multi-node orchestration), Nomad (HashiCorp)

> 💡 **Define verdict:** 3-tier stack (web + cache + db) declared with health-check dependency ordering — `service_healthy` prevents Flask/Postgres race condition.
> ➡️ Topology complete; proceed to Configure phase to externalize secrets.

---

## 2.1 · Environment & Secrets Management — Configure

Services need runtime configuration: database URLs, API keys, feature flags. Hardcoding these in `docker-compose.yml` creates three problems:

1. **Security**: Secrets committed to Git leak to anyone with repo access
2. **Environment drift**: Different values for dev/staging/prod require maintaining multiple YAML files
3. **Auditability**: Changing a password means editing YAML, redeploying, and hoping you didn't break syntax

**The solution**: Externalize configuration with `.env` files and Docker secrets.

### Phase 2 Code Snippet — .env File Configuration

Create `.env` in the same directory as `docker-compose.yml`:

```bash
# .env — never commit this file (add to .gitignore)
POSTGRES_USER=app_user
POSTGRES_PASSWORD=strong_random_password_here
POSTGRES_DB=production_db

# Flask configuration
FLASK_ENV=production
SECRET_KEY=flask_secret_key_change_this
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}
REDIS_URL=redis://cache:6379/0
```

Update `docker-compose.yml` to reference variables:

```yaml
services:
  web:
    build: .
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - SECRET_KEY=${SECRET_KEY}
    env_file:
      - .env  # Load all variables from .env

  db:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
    env_file:
      - .env
```

**Key details**:
- `${VARIABLE}` syntax interpolates .env values into YAML
- `.gitignore` should include `.env` — commit `.env.example` with placeholder values instead
- Production systems use Docker secrets or external vaults (see callout below)

> 💡 **Industry Standard:** Docker Secrets (Swarm/Kubernetes) or External Vaults
> ```yaml
> # Docker Swarm secrets (production)
> services:
>   db:
>     secrets:
>       - db_password
> secrets:
>   db_password:
>     external: true  # Managed outside Compose (e.g., AWS Secrets Manager)
> ```
> **When to use:** Production deployments. Never use .env files in production — they're unencrypted files on disk.
> **Common alternatives:** AWS Secrets Manager, HashiCorp Vault, Azure Key Vault, Kubernetes Secrets

> 💡 **Configure verdict:** Secrets moved from YAML to `.env` — same `docker-compose.yml` works across dev/staging/prod with different `.env` files.
> ➡️ Credentials out of git history; proceed to Orchestrate phase.

---

## 2.2 · Multi-Container Lifecycle Management — Orchestrate

You've defined services and configuration. Now control *how* they start, scale, and recover from failures.

Three orchestration levers:
1. **Startup ordering** — `depends_on` with health checks
2. **Restart policies** — automatic recovery from crashes
3. **Resource limits** — prevent one service from starving others

### Phase 3 Code Snippet — Dependency Ordering & Restart Policies

```yaml
services:
  web:
    build: .
    depends_on:
      db:
        condition: service_healthy  # CRITICAL: wait for health check
      cache:
        condition: service_started  # Just wait for container to start
    restart: unless-stopped  # Auto-restart on crash (but not if manually stopped)
    deploy:
      resources:
        limits:
          cpus: '1.0'      # Max 1 CPU core
          memory: 512M     # Max 512MB RAM
        reservations:
          cpus: '0.5'      # Guaranteed 0.5 cores
          memory: 256M

  db:
    image: postgres:16-alpine
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 10s  # Grace period before health checks start
    deploy:
      resources:
        limits:
          memory: 1G

  cache:
    image: redis:7-alpine
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 256M
```

**Key details**:
- `service_healthy` condition prevents web from crashing due to database not ready
- `unless-stopped` restarts containers on crash, but respects manual `docker compose stop`
- `start_period: 10s` gives Postgres time to initialize before failing health checks
- Resource limits prevent memory leaks from killing the host

### Scaling Services Horizontally

Run multiple replicas of stateless services:

```bash
# Scale web tier to 3 replicas (load balancer required — see Phase 4)
docker compose up -d --scale web=3

# Check running containers
docker compose ps
```

**Important**: Scaling exposes two limitations:
1. **Port conflicts**: If `web` exposes port 5000, all 3 replicas will fight for it — solution: use reverse proxy (Traefik, Nginx)
2. **Session state**: If Flask stores sessions in memory, users will be randomly logged out as requests hit different replicas — solution: use Redis for shared session storage (already in our stack!)

> 💡 **Industry Standard:** Traefik or Nginx for Load Balancing
> ```yaml
> services:
>   traefik:
>     image: traefik:v2.10
>     command:
>       - "--providers.docker=true"
>       - "--entrypoints.web.address=:80"
>     ports:
>       - "80:80"
>     volumes:
>       - /var/run/docker.sock:/var/run/docker.sock
>   web:
>     labels:
>       - "traefik.enable=true"
>       - "traefik.http.routers.web.rule=Host(`api.example.com`)"
> ```
> **When to use:** When scaling beyond 1 replica of a web service. Traefik auto-discovers containers via Docker socket.
> **Common alternatives:** Nginx, HAProxy, AWS ALB, Kubernetes Ingress

> 💡 **Orchestrate verdict:** Health-check ordering eliminates startup races; `unless-stopped` auto-heals crashes; resource limits cap memory at 1 GB per service.
> ➡️ Services self-heal; horizontal scaling via `--scale` ready for load balancer; proceed to Observe phase.

---

## 2.3 · Logs & Health Monitoring — Observe

Orchestration without observability is flying blind. When a service crashes at 3am, you need:
1. **Centralized logs** — view all services in one stream
2. **Health metrics** — current state of each service
3. **Service discovery** — inspect network topology

### Phase 4 Code Snippet — Health Checks & Log Aggregation

**View logs from all services**:
```bash
# Follow logs from all services (tail -f equivalent)
docker compose logs -f

# Filter to specific service
docker compose logs -f web

# Show last 100 lines from db service
docker compose logs --tail=100 db

# Include timestamps
docker compose logs -f --timestamps web
```

**Check service health status**:
```bash
# Show all services with health status
docker compose ps

# Output:
# NAME        IMAGE               STATUS                    PORTS
# web_1       flask-app:latest    Up 2 minutes (healthy)    0.0.0.0:5000->5000/tcp
# db_1        postgres:16-alpine  Up 2 minutes (healthy)    5432/tcp
# cache_1     redis:7-alpine      Up 2 minutes              6379/tcp
```

**Inspect individual container**:
```bash
# Get detailed health check results
docker inspect <container_id> | grep -A 20 Health

# Manual health check test
docker compose exec db pg_isready -U user
# Output: /var/run/postgresql:5432 - accepting connections
```

### Implementing Robust Health Checks

All stateful services need health checks. Here are production-grade examples:

```yaml
services:
  db:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s      # Check every 10 seconds
      timeout: 5s        # Fail if check takes >5s
      retries: 5         # Mark unhealthy after 5 consecutive failures
      start_period: 30s  # Grace period for slow startup (don't fail during init)

  cache:
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 3

  web:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 15s
```

**Critical**: The `web` service health check requires a `/health` endpoint in your Flask app:

```python
# app.py
@app.route('/health')
def health_check():
    """Health check endpoint for Docker Compose."""
    try:
        # Test database connection
        db.session.execute('SELECT 1')
        # Test Redis connection
        cache.ping()
        return {'status': 'healthy'}, 200
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}, 503
```

> 💡 **Industry Standard:** Structured Logging with ELK Stack or Loki
> ```yaml
> services:
>   web:
>     logging:
>       driver: "json-file"
>       options:
>         max-size: "10m"
>         max-file: "3"
>         labels: "service,environment"
>
>   # For production: send to centralized logging (Elasticsearch, Loki, CloudWatch)
>   loki:
>     image: grafana/loki:2.9.0
>     ports:
>       - "3100:3100"
> ```
> **When to use:** Production systems with >3 services. Local dev can use `docker compose logs`.
> **Common alternatives:** ELK Stack (Elasticsearch/Logstash/Kibana), Splunk, Datadog, AWS CloudWatch Logs

### Service Discovery & Debugging

Compose provides built-in DNS for service discovery:

```bash
# From inside web container, test connectivity
docker compose exec web sh
$ ping db           # Resolves to db container IP
$ ping cache        # Resolves to cache container IP
$ curl http://db:5432  # Test if port is reachable
```

**Common debugging commands**:
```bash
# Inspect network topology
docker network inspect <project>_app-network

# Check which containers are on the network
docker network inspect <project>_app-network | grep -A 5 Containers

# View environment variables in running container
docker compose exec web env

# Execute arbitrary commands in service containers
docker compose exec db psql -U user -d appdb -c "SELECT COUNT(*) FROM users;"
```

> 💡 **Observe verdict:** `docker compose ps` shows unhealthy services instantly; centralized logs surface root cause without `docker exec` in most cases.
> ➡️ Observability complete; chapter ready for Kubernetes orchestration in Ch.3.

---

## Step 4: Start the Entire Stack

```bash
docker compose up -d
```

**What happens**:
1. Compose creates `app-network` and `postgres-data` volume (if they don't exist).
2. Starts `cache` (Redis) — no dependencies.
3. Starts `db` (Postgres) — waits for health check to pass.
4. Starts `web` (Flask) — only after `db` is healthy and `cache` is running.

**Check logs**:
```bash
docker compose logs -f web
```

**Test the API**:
```bash
curl http://localhost:5000/users
```

**Tear down**:
```bash
docker compose down        # Stops containers, removes networks (keeps volumes)
docker compose down --volumes  # Also deletes postgres-data volume
```

---

## 3 · Mental Model — Services, Networks, Volumes as YAML Primitives

Think of Docker Compose as a **configuration-driven runtime** with three core primitives:

### Primitive 1: Services

A service is a container blueprint. You define:
- `image` or `build` (where to get the container)
- `ports` (expose to host)
- `environment` (config variables)
- `depends_on` (startup dependencies)
- `restart` policy (`no`, `always`, `on-failure`, `unless-stopped`)

**Scaling**: Run multiple replicas of a service:
```bash
docker compose up -d --scale web=3
```
This creates 3 `web` containers. If you exposed port 5000, you'd need a load balancer in front (see Ch.7).

### Primitive 2: Networks

By default, Compose creates a single network for all services. You can define multiple networks to isolate tiers:

```yaml
networks:
  frontend:  # Web tier only
  backend:   # Database tier only

services:
  web:
    networks:
      - frontend
      - backend
  db:
    networks:
      - backend  # Not visible to external clients
```

Services communicate via DNS: `http://web:5000`, `postgresql://db:5432`.

### Primitive 3: Volumes

Three types:
1. **Named volumes**: Managed by Docker, persistent across restarts (`postgres-data:/var/lib/postgresql/data`).
2. **Bind mounts**: Link host directories into containers (`./app:/app` — common for development live-reload).
3. **Anonymous volumes**: Temporary storage, deleted when container stops.

**Rule**: Use named volumes for production data (databases, uploaded files). Use bind mounts for development (live code changes without rebuilding images).

---

## 4 · What Can Go Wrong — Service Startup Ordering, Network Isolation, Port Conflicts

> **Debugging mindset**: When orchestration fails, the error appears in one layer (container crash, connection refused, data loss) but the root cause lives in another layer (YAML config, health check timing, volume lifecycle). The diagnostic flow: **symptom** → **hypothesis** → **validation** → **fix**. Each failure below demonstrates this 4-step pattern.

### Failure 1: Race Condition on Database Readiness

**The scenario**: Your Flask app connects to Postgres in `__init__`:
```python
db = SQLAlchemy(app)  # Crashes if Postgres isn't ready
```

**Symptom**: `web` container starts, crashes with `connection refused`, restarts, crashes again (crash loop).

**Hypothesis**: `depends_on: db` only waits for the Postgres *container* to start — not for the database server *inside* the container to accept connections. Postgres takes 3-5 seconds to initialize.

**Validation**: Check logs:
```bash
docker compose logs web
# Output: sqlalchemy.exc.OperationalError: could not connect to server: Connection refused
```

**Fix 1: Health check condition (recommended)**
```yaml
web:
  depends_on:
    db:
      condition: service_healthy  # Wait for health check to pass
```

**Fix 2: Retry logic in app code (defense in depth)**
```python
import time
from sqlalchemy import create_engine

def connect_with_retry(db_url, retries=5):
    for i in range(retries):
        try:
            engine = create_engine(db_url)
            engine.connect()
            return engine
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"DB not ready, retrying in 2s... ({e})")
            time.sleep(2)
```

> 💡 **Industry Practice:** Always implement both health checks AND retry logic
>
> **Why both?** Health checks prevent startup races in orchestration layer. Retry logic handles transient failures after startup (network blips, database restarts). Production systems need defense in depth.
>
> **When to use:** Health checks are mandatory for stateful services (databases, message queues). Retry logic is mandatory for all network clients.

### Failure 2: Port Conflict

**The scenario**: You already have a PostgreSQL instance running on your laptop (port 5432). When you run `docker compose up`, you get:
```
Error: bind: address already in use
```

**Hypothesis**: Two processes can't bind to the same port. Your host Postgres and the containerized Postgres both want 5432.

**Validation**: Check what's using the port:
```bash
# Windows
netstat -ano | findstr :5432

# Linux/Mac
lsof -i :5432
```

**Fix 1: Change the host port mapping**
```yaml
db:
  ports:
    - "5433:5432"  # Expose on host port 5433 instead
```
Now connect via `postgresql://user:pass@localhost:5433/appdb`.

**Fix 2: Don't expose the port at all (recommended for production)**
```yaml
db:
  # Remove ports section entirely
```
Services can still communicate via the internal `app-network`. Only expose ports that need external access (e.g., the `web` service).

> 💡 **Industry Practice:** Minimal Port Exposure
>
> **Security principle**: Only expose ports that require external access. Database ports should NEVER be exposed in production — only the web/API layer should be accessible from outside the Docker network.
>
> **Common pattern**:
> - `web`: Expose 5000 (needs external traffic)
> - `db`: No ports exposed (only web tier needs access)
> - `cache`: No ports exposed (only web tier needs access)

### Failure 3: Network Isolation Breaks External Access

**The scenario**: You define a custom network for `web` and `db`, but forget to expose `web`'s port:
```yaml
services:
  web:
    networks:
      - backend  # No external access
```

**Symptom**: `curl http://localhost:5000` times out — the container is running but unreachable.

**Hypothesis**: Services on custom networks are isolated from the host by default. You must explicitly publish ports.

**Validation**: Check network configuration:
```bash
docker compose ps  # Container is Up, but no port mapping shown
docker network inspect <project>_backend  # See which containers are on network
```

**Fix**: Explicitly publish ports:
```yaml
web:
  ports:
    - "5000:5000"  # Bind container port 5000 to host port 5000
  networks:
    - backend
```

Alternatively, use `network_mode: host` (Linux only) to share the host's network stack — but this disables network isolation (not recommended for production).

> 💡 **Industry Practice:** Multi-Network Isolation for Defense in Depth
> ```yaml
> networks:
>   frontend:  # Public-facing tier
>   backend:   # Internal services only
>
> services:
>   web:
>     networks:
>       - frontend  # Can receive external traffic
>       - backend   # Can talk to database
>   db:
>     networks:
>       - backend  # Isolated from external access
> ```
> **When to use:** Any production system with >2 tiers. Prevents lateral movement if web tier is compromised.

### Failure 4: Volumes Deleted by Accident

**The scenario**: You run `docker compose down` to restart your services, and all your database data is gone.

**Hypothesis**: Someone ran `docker compose down --volumes` (or `-v`), which deletes named volumes. The default `docker compose down` preserves volumes.

**Validation**: Check if volume still exists:
```bash
docker volume ls | grep postgres-data
# Empty output = volume was deleted
```

**Prevention**:
1. Always use named volumes for production data:
   ```yaml
   volumes:
     postgres-data:/var/lib/postgresql/data
   ```
2. Back up volumes before destructive operations:
   ```bash
   docker run --rm -v postgres-data:/data -v $(pwd):/backup \
     busybox tar czf /backup/postgres-backup.tar.gz /data
   ```
3. Use external volume management (cloud block storage, NFS) for critical data

> 💡 **Industry Practice:** Backup Automation for Persistent Volumes
> ```yaml
> services:
>   backup:
>     image: postgres:16-alpine
>     volumes:
>       - postgres-data:/data:ro  # Read-only mount
>       - ./backups:/backups
>     command: >
>       sh -c "while true; do
>         pg_dump -U user appdb | gzip > /backups/backup_$(date +%Y%m%d_%H%M%S).sql.gz;
>         sleep 86400;
>       done"
> ```
> **When to use:** Any stateful service. Automate daily backups, test restore procedures monthly.
> **Common alternatives:** AWS EBS snapshots, GCP Persistent Disk snapshots, Azure Backup, Velero (Kubernetes)

---

## 5 · Progress Check — Debug a Broken Compose File

> **Apply the 4-phase diagnostic workflow**: When a Compose stack fails, trace through DEFINE → CONFIGURE → ORCHESTRATE → OBSERVE to isolate the failure layer. Most failures appear in Phase 4 (logs/health) but originate in Phase 1-3 (YAML config, dependencies, resource limits).

**Scenario**: Your team gives you this `docker-compose.yml`:

```yaml
version: '3.9'

services:
  api:
    image: node:18
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://user:pass@db:5432/appdb
    command: npm start

  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: pass
```

You run `docker compose up` and see:
```
api_1 | Error: connect ECONNREFUSED db:5432
api_1 exited with code 1
```

**Question 1**: What's wrong with the service dependencies? (Phase 3: ORCHESTRATE)

<details>
<summary>Click to reveal</summary>

**Answer**: The `api` service has no `depends_on` clause. It starts immediately and tries to connect to Postgres before the `db` container is ready. The fix:

```yaml
api:
  depends_on:
    db:
      condition: service_healthy
```

Add a health check to `db`:
```yaml
db:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U user"]
    interval: 5s
    timeout: 3s
    retries: 5
```

**Phase violated**: Phase 3 (ORCHESTRATE) — startup ordering not defined.

</details>

**Question 2**: The database data disappears every time you run `docker compose down`. Why? (Phase 1: DEFINE)

<details>
<summary>Click to reveal</summary>

**Answer**: No named volume is defined for Postgres. By default, data is stored in an anonymous volume that gets deleted when the container is removed. Fix:

```yaml
db:
  volumes:
    - db-data:/var/lib/postgresql/data

volumes:
  db-data:
```

**Phase violated**: Phase 1 (DEFINE) — volume persistence not configured.

</details>

**Question 3**: You want to scale the `api` service to 3 replicas. What command do you run? (Phase 3: ORCHESTRATE)

<details>
<summary>Click to reveal</summary>

**Answer**:
```bash
docker compose up -d --scale api=3
```

**Caveat**: If you exposed port 3000, all 3 containers will conflict trying to bind to the same port. Solutions:
- Remove the `ports` section and use a reverse proxy (e.g., Nginx, Traefik) — proper Phase 1 architecture
- Use a port range: `"3000-3002:3000"` (limited scalability, not recommended)

**Phase consideration**: Horizontal scaling is Phase 3 (ORCHESTRATE), but load balancing is Phase 1 (DEFINE) architecture.

</details>

**Question 4**: The API crashes after 2 hours with "out of memory". How do you prevent this? (Phase 3: ORCHESTRATE)

<details>
<summary>Click to reveal</summary>

**Answer**: Add resource limits to prevent runaway memory consumption:

```yaml
api:
  deploy:
    resources:
      limits:
        memory: 512M  # Hard limit
      reservations:
        memory: 256M  # Guaranteed minimum
```

When memory hits 512M, the container is killed and restarted (if `restart: unless-stopped` is set).

**Phase applied**: Phase 3 (ORCHESTRATE) — resource limits prevent cascading failures.

**Diagnostic flow**:
1. **Phase 4 (OBSERVE)**: Notice crash in `docker compose logs api` — "JavaScript heap out of memory"
2. **Phase 3 (ORCHESTRATE)**: Add memory limits to prevent recurrence
3. **Phase 2 (CONFIGURE)**: Investigate if environment variables are causing memory leak (e.g., oversized cache config)

</details>

**Question 5**: After adding secrets to a `.env` file, the API still uses hardcoded credentials. Why? (Phase 2: CONFIGURE)

<details>
<summary>Click to reveal</summary>

**Answer**: The service must explicitly reference the .env file:

```yaml
api:
  env_file:
    - .env  # Load variables from .env
  environment:
    - DATABASE_URL=${DATABASE_URL}  # Interpolate from .env
```

**Common mistake**: Creating a .env file is not enough — services must declare `env_file` or explicit `environment` interpolation.

**Phase violated**: Phase 2 (CONFIGURE) — environment variable loading not declared.

**Validation**: Check loaded environment:
```bash
docker compose exec api env | grep DATABASE_URL
```

</details>

---

## 5.1 · Putting It Together — The Complete Decision Flow

```mermaid
flowchart TD
    Start([New Multi-Container Project]) --> Phase1[Phase 1: DEFINE]

    Phase1 --> Define1{Services defined?}
    Define1 -->|No| DefineServices[Declare services in YAML]
    DefineServices --> Define2{Networks isolated?}
    Define1 -->|Yes| Define2
    Define2 -->|No| DefineNetworks[Create custom networks]
    DefineNetworks --> Define3{Volumes persistent?}
    Define2 -->|Yes| Define3
    Define3 -->|No| DefineVolumes[Add named volumes]
    DefineVolumes --> Phase2[Phase 2: CONFIGURE]
    Define3 -->|Yes| Phase2

    Phase2 --> Config1{Secrets externalized?}
    Config1 -->|No| ConfigEnv[Move to .env file]
    ConfigEnv --> Config2{Variables interpolated?}
    Config1 -->|Yes| Config2
    Config2 -->|No| ConfigInterpolate[Use ${VAR} syntax]
    ConfigInterpolate --> Phase3[Phase 3: ORCHESTRATE]
    Config2 -->|Yes| Phase3

    Phase3 --> Orch1{Dependencies ordered?}
    Orch1 -->|No| OrchDepends[Add depends_on + health checks]
    OrchDepends --> Orch2{Restart policy set?}
    Orch1 -->|Yes| Orch2
    Orch2 -->|No| OrchRestart[Set restart: unless-stopped]
    OrchRestart --> Orch3{Resource limits?}
    Orch2 -->|Yes| Orch3
    Orch3 -->|No| OrchLimits[Add CPU/memory limits]
    OrchLimits --> StartStack[docker compose up -d]
    Orch3 -->|Yes| StartStack

    StartStack --> Phase4[Phase 4: OBSERVE]

    Phase4 --> Obs1{Services healthy?}
    Obs1 -->|No| ObsLogs[Check: docker compose logs]
    ObsLogs --> ObsHealth[Check: docker compose ps]
    ObsHealth --> ObsDebug{Root cause?}
    ObsDebug -->|Config error| Phase2
    ObsDebug -->|Dependency error| Phase3
    ObsDebug -->|Architecture error| Phase1
    Obs1 -->|Yes| Obs2{Logs aggregated?}
    Obs2 -->|No| ObsLogging[Set up centralized logging]
    ObsLogging --> Obs3{Scaling needed?}
    Obs2 -->|Yes| Obs3
    Obs3 -->|No| Production[Ready for Production]
    Obs3 -->|Yes| ObsScale[Add load balancer + scale replicas]
    ObsScale --> Production

    Production --> End([Deployed Stack])

    style Phase1 fill:#1e3a8a,color:#fff
    style Phase2 fill:#15803d,color:#fff
    style Phase3 fill:#b45309,color:#fff
    style Phase4 fill:#1d4ed8,color:#fff
    style Production fill:#15803d,color:#fff
    style End fill:#22c55e,color:#fff
```

> **Reading the flowchart**: Each phase has decision gates that must pass before proceeding to the next phase. When a failure occurs in Phase 4 (OBSERVE), trace back to the originating phase based on error type. **Configuration errors** → Phase 2. **Dependency/ordering errors** → Phase 3. **Architecture errors** (wrong network topology, missing volumes) → Phase 1.

### Real-World Scenario — Complete 4-Phase Walkthrough

**The mission**: Deploy a production e-commerce stack: Nginx (web server) → Node.js API (application) → PostgreSQL (database) → Redis (session cache).

**Phase 1 (DEFINE)**: 3 services defined → DB fails health check → Check logs: port conflict → Fix port mapping → All services healthy → **DECISION: Scale web to 3 replicas**

**Phase 2 (CONFIGURE)**: Move DATABASE_URL to .env → API still crashes → Check env with `docker compose exec api env` → Missing env_file declaration → Add env_file → API connects successfully → **DECISION: Rotate credentials monthly**

**Phase 3 (ORCHESTRATE)**: API starts before DB → Add depends_on: service_healthy → API stable → Memory leak after 6 hours → Add memory limit (1GB) → Container auto-restarts → **DECISION: Investigate memory leak in app code**

**Phase 4 (OBSERVE)**: Check docker compose logs → 1000 lines/minute overwhelming terminal → Set up centralized logging (Loki) → Filter by service → **DECISION: Alert on error rate >10/min**

**Outcome**: Production-ready stack with health monitoring, persistent storage, auto-restart, and centralized observability.

---

## 6 · Bridge to Ch.3 — Compose Works on One Machine; Kubernetes Scales Across Many

**What you've learned**: Docker Compose orchestrates multi-container applications with declarative service definitions, health checks, and persistent volumes. You've mastered the **4-phase practitioner workflow**:

| Phase | What You Learned | Limitation on Single Machine |
|-------|------------------|------------------------------|
| **Phase 1: DEFINE** | Service topology, networks, volumes | No cross-machine networking |
| **Phase 2: CONFIGURE** | .env files, secrets externalization | No encrypted secret stores |
| **Phase 3: ORCHESTRATE** | Health checks, restart policies, scaling | Scale limited to host capacity |
| **Phase 4: OBSERVE** | Logs, health monitoring, DNS discovery | No cluster-wide metrics |

You can run a production-grade 3-tier stack (web + cache + database) locally with one command, but Compose hits fundamental limits when you need:

**What Compose can't do**:
- **Horizontal scaling across machines**: `docker compose up --scale web=10` runs 10 containers *on your laptop*. Production systems need to distribute load across multiple nodes.
- **Self-healing across nodes**: If a container crashes, Compose restarts it (with `restart: always`) — but if the *host machine* fails, your entire stack goes down. No automatic migration to healthy nodes.
- **Rolling updates**: Deploying a new version requires `docker compose down && docker compose up` — causing downtime. Zero-downtime deployments need orchestration across multiple containers.
- **Load balancing**: Scaling a service creates multiple containers, but Compose doesn't automatically route traffic between them. You need an external load balancer (Traefik, Nginx).

**Enter Kubernetes**: The next chapter introduces Kubernetes (K8s) — the industry-standard orchestrator for multi-node clusters. You'll learn:
- **Pods**: The Kubernetes equivalent of a Compose service — but pods can migrate between nodes
- **Deployments**: Declarative updates with zero downtime (rolling updates, blue-green deployments)
- **Services**: Built-in load balancing for scaled applications with automatic endpoint discovery
- **Kind (Kubernetes in Docker)**: Run a full K8s cluster locally for free

**The mental models transfer directly**:

| Docker Compose Concept | Kubernetes Equivalent | New Capabilities |
|------------------------|----------------------|------------------|
| Service | Pod + Deployment | Multi-node scheduling, auto-scaling |
| Network | Service + NetworkPolicy | Cluster-wide DNS, ingress controllers |
| Volume | PersistentVolume | Cloud-native block storage, CSI drivers |
| Health check | Readiness/Liveness probes | Automated pod restarts, traffic routing |
| depends_on | Init containers | Dependency management at pod level |
| docker compose up | kubectl apply | Declarative state management |

**The 4-phase workflow scales to Kubernetes**:
- **Phase 1 (DEFINE)** → Write Kubernetes manifests (YAML) instead of docker-compose.yml
- **Phase 2 (CONFIGURE)** → Use Kubernetes Secrets and ConfigMaps instead of .env files
- **Phase 3 (ORCHESTRATE)** → Deployments handle rolling updates, ReplicaSets handle scaling
- **Phase 4 (OBSERVE)** → Kubernetes Dashboard, Prometheus, Grafana for cluster-wide observability

**For now**: Use Docker Compose for local development, testing, and small production deployments (<10 containers on a single machine). When you need to scale beyond one node, graduate to Kubernetes (Ch.3).

**The progression**:
- **Ch.1 (Docker)**: Containerize one application
- **Ch.2 (Compose)**: Orchestrate multiple containers on one machine ← *you are here*
- **Ch.3 (Kubernetes)**: Orchestrate across many machines

---

## What's Next

- **Local**: Run `notebook.ipynb_solution.ipynb` (reference) or `notebook.ipynb_exercise.ipynb` (practice) to build a Flask + Postgres + Redis stack from scratch
- **Azure** (optional): Run `notebook_supplement.ipynb_solution.ipynb` (reference) or `notebook_supplement.ipynb_exercise.ipynb` (practice) to deploy multi-container groups to Azure Container Instances (ACI)
- **Ch.3**: Learn Kubernetes fundamentals with Kind (no cloud account required)
