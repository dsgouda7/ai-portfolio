# Ch.6 — Infrastructure as Code

> **The story.** In **2006**, **Amazon Web Services** launched **EC2**, making server provisioning as simple as an API call — but engineers still managed infrastructure through web consoles, clicking through forms to configure load balancers and databases. The breakthrough came in **2010** when **Mitchell Hashimoto** (then a student at University of Washington) founded **HashiCorp** and released **Vagrant** to automate VM provisioning. But the tool that revolutionized infrastructure was **Terraform**, released in **2014** by Hashimoto's team as an open-source Infrastructure as Code (IaC) framework. Terraform introduced a *declarative* model — you describe the desired end state (5 web servers, 1 database, 2 load balancers), and Terraform calculates the diff and applies changes idempotently. Unlike imperative scripts that break if run twice, Terraform's state file tracks reality and makes infrastructure reproducible. By 2024, Terraform manages infrastructure for 90% of Fortune 500 companies — every Kubernetes cluster, S3 bucket, and VPC you provision likely started as a `.tf` file in a Git repo.
>
> **Where you are in the curriculum.** You've containerized apps with Docker (Ch.1), orchestrated multi-container stacks with Docker Compose (Ch.2), deployed to Kubernetes for auto-healing (Ch.3), automated CI/CD with GitHub Actions (Ch.4), and instrumented observability with Prometheus + Grafana (Ch.5). You can ship code to production — **but you're provisioning infrastructure manually**. If you need to spin up a new environment (staging, DR failover, customer demo), you're clicking through AWS consoles or running ad-hoc shell scripts. Changes drift over time — dev and prod environments diverge because someone forgot to document a security group rule. This chapter gives you **Infrastructure as Code** — the foundation for reproducible, version-controlled, peer-reviewed infrastructure deployments.
>
> **Notation in this chapter.** **Terraform** — open-source IaC tool; **HCL** — HashiCorp Configuration Language (the `.tf` syntax); **Provider** — plugin that talks to APIs (AWS, Azure, Docker, Kubernetes); **Resource** — infrastructure component (EC2 instance, S3 bucket, Docker container); **State** — Terraform's record of managed infrastructure (`terraform.tfstate`); **Plan** — preview of changes before applying them (`terraform plan`); **Apply** — execute changes to match desired state (`terraform apply`); **Drift** — when real infrastructure diverges from code (manual changes outside Terraform).

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Provision a Docker container running Nginx using Terraform — infrastructure defined as code, version-controlled, and reproducible across environments.

**What we know so far:**
- ✅ We can run containers with Docker (Ch.1)
- ✅ We can orchestrate multi-container apps with Docker Compose (Ch.2)
- ✅ We can deploy to Kubernetes for production (Ch.3)
- ❌ **But infrastructure provisioning is still manual and error-prone!**

**What's blocking us:**
We're managing infrastructure through ad-hoc scripts and manual steps:
- **No version control**: Infrastructure changes aren't tracked in Git
- **No peer review**: Can't review infrastructure changes like code PRs
- **No rollback**: Can't revert to a previous infrastructure state
- **No reproducibility**: Can't spin up identical environments (dev, staging, prod)
- **Drift**: Manual changes bypass documentation and cause config drift

Without Infrastructure as Code, you can't treat infrastructure with the same rigor as application code — no testing, no code reviews, no rollback safety nets.

**What this chapter unlocks:**
The **Terraform workflow** — define infrastructure as declarative `.tf` files, version-control them in Git, preview changes with `terraform plan`, apply changes with `terraform apply`, and destroy environments with `terraform destroy`.
- **Establishes the IaC foundation**: Resources, providers, state management
- **Provides concrete examples**: Provision Docker containers, networks, volumes
- **Teaches production workflows**: State backends, variable management, drift detection

✅ **This is the foundation** — every later chapter assumes infrastructure is version-controlled and reproducible.

---

## Animation

![Chapter animation](img/ch06-terraform-workflow.gif)

## 1 · Infrastructure as Code Means You Declare Desired State and Let Terraform Calculate the Diff

Traditional infrastructure scripts are *imperative* — they tell the system step-by-step what to do:
```bash
# Imperative approach — breaks if run twice!
aws ec2 run-instances --image-id ami-12345 --count 2
aws elb create-load-balancer --name my-lb
aws elb register-instances --lb my-lb --instances i-abc123 i-def456
```

If you run this script twice, you get 4 instances and 2 load balancers. You need custom logic to check "does this already exist?" before creating.

Infrastructure as Code is *declarative* — you describe the end state, and Terraform figures out how to get there:
```hcl
resource "aws_instance" "web" {
  count         = 2
  ami           = "ami-12345"
  instance_type = "t2.micro"
}

resource "aws_lb" "main" {
  name = "my-lb"
}
```

Run `terraform apply` once: creates 2 instances + 1 load balancer.  
Run `terraform apply` again: **no changes** (already matches desired state).  
Change `count = 3`: Terraform adds 1 instance (doesn't destroy and recreate everything).

**Key insight:** Terraform maintains a **state file** (`terraform.tfstate`) that records what it's managing. Every `terraform apply` compares *desired state* (your `.tf` files) against *current state* (the state file + reality check) and applies only the necessary changes.

---

## 1.5 · The Practitioner Workflow — Your 5-Phase Infrastructure Provisioning Journey

> ⚠️ **Two ways to read this chapter:**
> - **Theory-first (recommended for learning):** Read §0→§4 sequentially to understand the concepts, then use this workflow as your reference
> - **Workflow-first (practitioners with existing knowledge):** Use this diagram as a jump-to guide when working with real infrastructure
>
> **Note:** Section numbers don't follow phase order because the chapter teaches concepts pedagogically (theory before application). The workflow below shows how to APPLY those concepts.

**What you'll build by the end:** A complete Infrastructure as Code workflow managing production resources — from writing Terraform modules, through previewing and applying changes, to validating deployments and maintaining state over time. This is the foundation every DevOps engineer uses to manage infrastructure at scale.

```
Phase 1: DEFINE           Phase 2: PLAN              Phase 3: APPLY            Phase 4: VALIDATE         Phase 5: MAINTAIN
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Write Terraform modules:  Preview changes:           Provision infrastructure:  Verify resources:         Manage state & drift:

• Configure providers     • Run terraform plan       • Execute terraform apply  • Test connectivity       • Configure remote backend
• Define resources        • Review diff output       • Monitor progress         • Check security groups   • Enable state locking
• Declare variables       • Estimate costs           • Handle errors            • Validate compliance     • Detect drift
• Set outputs             • Get approval             • Update state file        • Run smoke tests         • Plan rollback strategy

→ DECISION:               → DECISION:                → DECISION:                → DECISION:               → DECISION:
  What to provision?        Apply or abort?            Rollback on error?         Pass or fix?              State backend choice?
  • VPC + subnets           • +12 resources            • Error → investigate      • SSH fails → fix SG      • Local: dev/testing
  • Security groups         • Cost: $145/mo OK         • Success → validate       • HTTP 200 → success      • S3+DynamoDB: prod
  • Compute instances       • Breaking change? No      • Partial fail → cleanup   • DNS resolves → ready    • Terraform Cloud: team
  • Load balancers          • Approved → apply                                                              • State lock prevents
  • Databases                                                                                                   concurrent changes
```

**The workflow maps to these sections:**
- **Phase 1 (Define)** → §3 Terraform Workflow (init, write configs)
- **Phase 2 (Plan)** → §4.2 State Reconciliation, §3 (terraform plan)
- **Phase 3 (Apply)** → §4.1 Dependency Graph, §3 (terraform apply)
- **Phase 4 (Validate)** → §5 What Can Go Wrong
- **Phase 5 (Maintain)** → §4.3 Drift Detection, §5.1 State File Management

> 💡 **How to use this workflow:** Complete Phase 1→2→3→4→5 in order on your infrastructure. The sections above teach WHY each phase works; refer back here for WHAT to do at each step.

### Phase 1: DEFINE — Write Terraform Configuration

**Entry criteria:** You know what infrastructure you need to provision (e.g., web app with database, load balancer, monitoring).

**Goal:** Write `.tf` files defining all resources, their configuration, and dependencies.

**Key decisions at this phase:**
1. **Which cloud provider?** (AWS, Azure, GCP, Docker, Kubernetes)
2. **What resources are required?** (compute, networking, storage, security)
3. **How should resources be parameterized?** (variables for reuse across environments)
4. **What information should be exported?** (outputs for downstream automation)

**The anatomy of a Terraform module:**

```hcl
# providers.tf — declare provider and version
terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
  required_version = ">= 1.5"
}

provider "docker" {
  host = "unix:///var/run/docker.sock"  # Local Docker daemon
}

# main.tf — define resources
resource "docker_network" "private_network" {
  name = "${var.environment}-network"
}

resource "docker_container" "web" {
  name  = "${var.environment}-nginx"
  image = "nginx:${var.nginx_version}"
  
  ports {
    internal = 80
    external = var.external_port
  }
  
  networks_advanced {
    name = docker_network.private_network.name
  }
  
  volumes {
    host_path      = "${path.cwd}/html"
    container_path = "/usr/share/nginx/html"
  }
}

# variables.tf — parameterize configuration
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "nginx_version" {
  description = "Nginx Docker image tag"
  type        = string
  default     = "1.25"
}

variable "external_port" {
  description = "Port to expose on host"
  type        = number
  default     = 8080
}

# outputs.tf — export values for downstream use
output "container_id" {
  description = "Docker container ID"
  value       = docker_container.web.id
}

output "container_ip" {
  description = "Container IP address"
  value       = docker_container.web.network_data[0].ip_address
}

output "access_url" {
  description = "URL to access the service"
  value       = "http://localhost:${var.external_port}"
}
```

**Common mistakes at this phase:**
- ❌ Hardcoding values instead of using variables (prevents reuse across environments)
- ❌ Not declaring provider version constraints (leads to version drift)
- ❌ Defining all resources in one file (monolithic, hard to maintain)
- ❌ Missing dependencies (Terraform creates resources in wrong order)

**Exit criteria:** You have valid `.tf` files that pass `terraform validate`. Ready for Phase 2 (preview changes).

> 💡 **Industry callout #1 — Terraform vs Pulumi vs CloudFormation:** Terraform (HCL syntax, 150k+ modules, multi-cloud) dominates 70% of IaC market; Pulumi (Python/TypeScript/Go, full programming language) growing 40% YoY for teams wanting type safety + loops; CloudFormation (JSON/YAML, AWS-only, free) still used by 40% of AWS-native teams. Key tradeoffs: **Terraform = declarative simplicity + vast ecosystem**, **Pulumi = programming language expressiveness + testing**, **CloudFormation = zero-cost AWS lock-in**. For multi-cloud, Terraform wins. For complex logic, Pulumi wins. For AWS-only simplicity, CloudFormation wins.

---

### Phase 2: PLAN — Preview Changes Before Applying

**Entry criteria:** You have valid Terraform configuration (Phase 1 complete). State file exists (or will be created on first run).

**Goal:** Generate a detailed preview showing exactly what Terraform will create, update, or destroy. Review for correctness and cost impact before executing.

**Key decisions at this phase:**
1. **Are the changes correct?** (did you specify the right resources?)
2. **What's the blast radius?** (how many resources affected?)
3. **What's the cost impact?** (use Infracost or cloud pricing calculators)
4. **Are there breaking changes?** (e.g., recreating a database drops all data)
5. **Who needs to approve?** (production changes require peer review)

**Running terraform plan:**

```bash
# Initialize Terraform (downloads provider plugins)
$ terraform init

Initializing the backend...
Initializing provider plugins...
- Finding kreuzwerker/docker versions matching "~> 3.0"...
- Installing kreuzwerker/docker v3.0.2...
Terraform has been successfully initialized!

# Preview changes (no side effects — safe to run repeatedly)
$ terraform plan

Terraform will perform the following actions:

  # docker_container.web will be created
  + resource "docker_container" "web" {
      + attach            = false
      + id                = (known after apply)
      + image             = "nginx:1.25"
      + name              = "dev-nginx"
      + ports {
          + external = 8080
          + internal = 80
          + protocol = "tcp"
        }
    }

  # docker_network.private_network will be created
  + resource "docker_network" "private_network" {
      + driver      = "bridge"
      + id          = (known after apply)
      + name        = "dev-network"
    }

Plan: 2 to add, 0 to change, 0 to destroy.

Changes to Outputs:
  + access_url   = "http://localhost:8080"
  + container_id = (known after apply)
  + container_ip = (known after apply)
```

**Reading the plan output:**
- `+` symbol → resource will be **created**
- `~` symbol → resource will be **updated in-place**
- `-/+` symbol → resource will be **destroyed and recreated** (breaking change!)
- `-` symbol → resource will be **destroyed**
- `(known after apply)` → value computed by provider after creation

**Cost estimation with Infracost:**

```bash
# Install Infracost (optional but recommended for production)
$ brew install infracost  # or download from infracost.io

# Generate cost estimate from Terraform plan
$ infracost breakdown --path .

Project: example-infrastructure

 Name                                   Monthly Qty  Unit    Monthly Cost
 
 aws_instance.web[0]                                                     
 ├─ Instance usage (Linux, on-demand)          730  hours         $7.30
 └─ EBS storage (gp3)                            30  GB            $2.40
 
 aws_instance.web[1]                                                     
 ├─ Instance usage (Linux, on-demand)          730  hours         $7.30
 └─ EBS storage (gp3)                            30  GB            $2.40
 
 aws_lb.main                                                             
 ├─ Load balancer usage                         730  hours        $16.20
 └─ Load balancer data processed                100  GB            $0.80
 
 aws_db_instance.postgres                                                
 ├─ Database instance (db.t3.medium)            730  hours        $60.74
 └─ Storage (gp3, 100 GB)                       100  GB           $11.50
 
 OVERALL TOTAL                                                   $116.64

──────────────────────────────────
12 cloud resources detected:
∙ 12 were estimated, all of which include usage-based costs
```

**Decision checkpoint #1 — Plan review before apply:**

```
Scenario: You run terraform plan for a new VPC setup

Plan output shows:
  + 1 VPC
  + 3 subnets (public, private-app, private-db)
  + 4 security groups (ALB, web, db, bastion)
  + 2 EC2 instances (web servers)
  + 1 RDS instance (PostgreSQL)
  + 1 Application Load Balancer
  
  Plan: 12 to add, 0 to change, 0 to destroy
  
Infracost estimate: $145/month
  - EC2: $14.60
  - RDS: $60.74
  - ALB: $16.20
  - Data transfer: est. $15/mo
  - Remaining: EBS, snapshots, backups

Review checklist:
  ✅ All resources match design doc?          YES (checked against architecture diagram)
  ✅ Security groups follow least privilege?  YES (reviewed rules, no 0.0.0.0/0 on SSH)
  ✅ Cost within budget?                      YES ($145 < $200 budget)
  ✅ Breaking changes flagged?                NO (-/+ symbols would indicate recreation)
  ✅ Peer review approved?                    YES (pull request approved by 2 engineers)
  
DECISION: APPROVED → Proceed to terraform apply
```

**Exit criteria:** Plan reviewed and approved. No unexpected resources in diff. Cost estimated and within budget. Ready for Phase 3 (apply changes).

> 💡 **Industry callout #2 — Infracost for cost estimation:** Infracost (open-source, 15k+ GitHub stars) parses Terraform plans and displays cloud costs *before* you provision — essential for production where surprise $10k/month bills destroy budgets. Integrates with GitHub Actions to block PRs exceeding cost thresholds. Alternative: **CloudCost** (real-time spend tracking) and **Terraform Cloud Cost Estimation** (native, requires paid account). Best practice: run Infracost in CI on every PR targeting production infrastructure.

---

### Phase 3: APPLY — Provision Infrastructure

**Entry criteria:** Plan reviewed and approved (Phase 2 complete). You're ready to execute changes.

**Goal:** Run `terraform apply` to provision resources. Monitor progress, handle errors, update state file.

**Key decisions at this phase:**
1. **Execute immediately or require manual confirmation?** (`-auto-approve` for CI, interactive for production)
2. **Handle errors: rollback or investigate?** (depends on error type)
3. **Verify state file updated correctly?** (check `terraform.tfstate` after apply)

**Executing terraform apply:**

```bash
# Apply changes (interactive — requires confirmation)
$ terraform apply

# ... (plan output shown again) ...

Do you want to perform these actions?
  Terraform will perform the actions described above.
  Only 'yes' will be accepted to approve.

  Enter a value: yes

docker_network.private_network: Creating...
docker_network.private_network: Creation complete after 1s [id=abc123]
docker_container.web: Creating...
docker_container.web: Creation complete after 3s [id=def456]

Apply complete! Resources: 2 added, 0 changed, 0 destroyed.

Outputs:

access_url = "http://localhost:8080"
container_id = "def456789abc"
container_ip = "172.17.0.2"
```

**Understanding the apply output:**
- **Creating...** → Terraform is calling provider APIs to create resource
- **Creation complete after Xs** → Resource successfully created
- **Resources: 2 added, 0 changed, 0 destroyed** → Summary of changes
- **Outputs:** → Exported values from `outputs.tf`

**State file after apply:**

```bash
$ terraform show  # View current state in human-readable format

# docker_container.web:
resource "docker_container" "web" {
    attach            = false
    command           = [
        "nginx",
        "-g",
        "daemon off;",
    ]
    id                = "def456789abc"
    image             = "nginx:1.25"
    name              = "dev-nginx"
    # ... (full resource attributes)
}

# docker_network.private_network:
resource "docker_network" "private_network" {
    driver = "bridge"
    id     = "abc123456def"
    name   = "dev-network"
    # ... (full network attributes)
}
```

**Handling errors during apply:**

**Error scenario 1 — Resource already exists:**
```bash
Error: Error creating container: Conflict. The container name "/dev-nginx" is already in use

Solution:
  1. Check if resource was manually created outside Terraform
  2. Import existing resource: terraform import docker_container.web $(docker ps -aqf "name=dev-nginx")
  3. Or destroy manual resource and retry: docker rm -f dev-nginx && terraform apply
```

**Error scenario 2 — Provider authentication failed:**
```bash
Error: Error creating instance: AuthFailure: AWS credentials not configured

Solution:
  1. Set AWS credentials: export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
  2. Or configure AWS CLI: aws configure
  3. Or use IAM role (preferred for EC2/Lambda execution)
```

**Error scenario 3 — Resource limit exceeded:**
```bash
Error: Error creating VPC: VpcLimitExceeded (max 5 VPCs per region)

Solution:
  1. Delete unused VPCs in AWS console
  2. Or request limit increase from AWS Support
  3. Or use existing VPC (data source instead of resource)
```

**Decision checkpoint #2 — Error handling during apply:**

```
Scenario: terraform apply partially succeeds

Progress:
  ✅ docker_network.private_network: Created
  ✅ docker_container.db: Created
  ❌ docker_container.web: Error — port 8080 already in use
  ⏸️  docker_container.cache: Not attempted (depends on web)

Current state:
  - State file updated with network + db
  - Web container NOT in state (creation failed)
  - Cache container NOT in state (blocked by web failure)

DECISION TREE:
  Option A: Fix error and retry
    → Investigate: lsof -i :8080  # Find process using port 8080
    → Kill conflicting process or change external_port variable
    → Run: terraform apply  # Resumes from where it stopped
    → Result: Creates web + cache, preserves network + db
  
  Option B: Rollback all changes
    → Run: terraform destroy
    → Fix root cause in configuration
    → Run: terraform apply (clean start)
    → Result: All resources created consistently
  
  Recommended: OPTION A (partial state is valid, just fix and continue)
```

**Exit criteria:** All resources successfully created. State file updated. Outputs displayed. Ready for Phase 4 (validate deployment).

> 💡 **Industry callout #3 — Atlantis for PR-based Terraform workflows:** Atlantis (open-source, 7k+ stars) automates terraform plan/apply in GitHub/GitLab PRs — engineers open a PR with infrastructure changes, Atlantis comments with the plan output, reviewers approve, Atlantis runs apply on merge. Prevents "works on my machine" drift (all applies run from CI with shared state). Alternative: **Terraform Cloud** (SaaS, native HCP integration) and **Spacelift** (advanced policy engine). Best practice: use Atlantis for teams >5 engineers managing shared infrastructure.

---

### Phase 4: VALIDATE — Verify Infrastructure Health

**Entry criteria:** Terraform apply succeeded (Phase 3 complete). Resources are running but not yet verified.

**Goal:** Test connectivity, security, compliance. Ensure infrastructure works as designed.

**Key decisions at this phase:**
1. **What to test?** (connectivity, security, performance, compliance)
2. **Pass/fail criteria?** (HTTP 200, SSH accessible, database reachable)
3. **Fix or rollback on failure?** (depends on severity)

**Validation test suite:**

```bash
#!/bin/bash
# validate_infrastructure.sh — Post-apply smoke tests

set -e  # Exit on first error

echo "=== Phase 4: Infrastructure Validation ==="

# Extract outputs from Terraform
CONTAINER_IP=$(terraform output -raw container_ip)
ACCESS_URL=$(terraform output -raw access_url)
CONTAINER_ID=$(terraform output -raw container_id)

echo "Testing container: $CONTAINER_ID"
echo "Container IP: $CONTAINER_IP"
echo "Access URL: $ACCESS_URL"

# Test 1: Container is running
echo -n "✓ Container running? "
if docker ps --filter "id=$CONTAINER_ID" --format "{{.Status}}" | grep -q "Up"; then
  echo "PASS"
else
  echo "FAIL — container not running"
  exit 1
fi

# Test 2: HTTP endpoint responds
echo -n "✓ HTTP endpoint reachable? "
if curl -sf "$ACCESS_URL" > /dev/null; then
  echo "PASS"
else
  echo "FAIL — HTTP endpoint unreachable"
  exit 1
fi

# Test 3: Container logs show no errors
echo -n "✓ No errors in container logs? "
if docker logs "$CONTAINER_ID" 2>&1 | grep -qi "error"; then
  echo "FAIL — errors found in logs"
  docker logs "$CONTAINER_ID" | grep -i error
  exit 1
else
  echo "PASS"
fi

# Test 4: Network connectivity (container can reach external DNS)
echo -n "✓ Container has network connectivity? "
if docker exec "$CONTAINER_ID" ping -c 1 8.8.8.8 > /dev/null 2>&1; then
  echo "PASS"
else
  echo "FAIL — no external network connectivity"
  exit 1
fi

# Test 5: Expected ports are exposed
echo -n "✓ Port 8080 exposed? "
if docker port "$CONTAINER_ID" | grep -q "8080"; then
  echo "PASS"
else
  echo "FAIL — port 8080 not exposed"
  exit 1
fi

echo "=== All validation tests passed! ==="
```

**Decision checkpoint #3 — Validation failure troubleshooting:**

```
Scenario: HTTP endpoint test fails after terraform apply

Test output:
  ✓ Container running? PASS
  ✗ HTTP endpoint reachable? FAIL — connection refused

Investigation steps:
  1. Check if container is listening on expected port
     $ docker exec <container_id> netstat -tuln | grep 80
     tcp  0  0  0.0.0.0:80  0.0.0.0:*  LISTEN  ← Container listening on port 80 ✅
  
  2. Check if port mapping is correct
     $ docker port <container_id>
     (empty output)  ← Port NOT mapped! ❌
  
  3. Check Terraform configuration
     resource "docker_container" "web" {
       # ... ports block MISSING!
     }
  
Root cause: Forgot to add ports block in Terraform config

DECISION: FIX AND REAPPLY
  1. Add ports block to main.tf:
     ports {
       internal = 80
       external = 8080
     }
  2. Run: terraform plan  # Shows update in-place
  3. Run: terraform apply
  4. Rerun validation: ./validate_infrastructure.sh  ✅ PASS
```

**Compliance validation (production-critical):**

For production infrastructure, add compliance checks:

```bash
# Security validation with Checkov
$ pip install checkov
$ checkov --directory . --framework terraform

Check: CKV_AWS_23: "Ensure security group does not allow ingress from 0.0.0.0:0 to port 22"
  PASSED for resource: aws_security_group.web
  File: /main.tf:45-60

Check: CKV_AWS_8: "Ensure RDS instances have encryption at rest enabled"
  FAILED for resource: aws_db_instance.main
  File: /main.tf:120-135
  Fix: Add `storage_encrypted = true`

Summary: 34 passed, 2 failed, 0 skipped
```

**Exit criteria:** All validation tests pass. Infrastructure is healthy and meets compliance requirements. Ready for Phase 5 (ongoing maintenance).

> 💡 **Industry callout #4 — Checkov for Terraform security scanning:** Checkov (open-source by Bridgecrew, 6k+ stars) scans Terraform/CloudFormation for 1000+ security misconfigurations — unencrypted databases, overly permissive security groups, public S3 buckets. Runs in CI to block insecure infrastructure before apply. Alternative: **tfsec** (lightweight, 6k+ policies) and **Terraform Sentinel** (HashiCorp native, policy-as-code). Best practice: run Checkov in pre-commit hooks AND CI — catch issues before they reach production.

---

### Phase 5: MAINTAIN — State Management & Drift Detection

**Entry criteria:** Infrastructure is running and validated (Phase 4 complete). Now managing ongoing changes and preventing drift.

**Goal:** Configure remote state backend, enable state locking, detect manual changes, plan rollback strategy.

**Key decisions at this phase:**
1. **Where to store state?** (local for dev, remote backend for production)
2. **How to prevent concurrent applies?** (state locking with DynamoDB/Azure Blob/GCS)
3. **How often to check for drift?** (daily cron job, or on every plan)
4. **Rollback strategy?** (Git revert + terraform apply, or restore from state backup)

**Configuring remote state backend (S3 + DynamoDB):**

```hcl
# backend.tf — remote state configuration
terraform {
  backend "s3" {
    bucket         = "my-company-terraform-state"
    key            = "infrastructure/prod/terraform.tfstate"
    region         = "us-west-2"
    
    # State locking to prevent concurrent applies
    dynamodb_table = "terraform-state-lock"
    
    # Encryption at rest
    encrypt        = true
    
    # Versioning for rollback
    # (enable on S3 bucket separately)
  }
}
```

**Creating the S3 bucket and DynamoDB table:**

```bash
# One-time setup (run manually or via separate Terraform config)
$ aws s3api create-bucket \
    --bucket my-company-terraform-state \
    --region us-west-2 \
    --create-bucket-configuration LocationConstraint=us-west-2

$ aws s3api put-bucket-versioning \
    --bucket my-company-terraform-state \
    --versioning-configuration Status=Enabled

$ aws dynamodb create-table \
    --table-name terraform-state-lock \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-west-2

# Migrate local state to remote backend
$ terraform init -migrate-state
```

**Drift detection workflow:**

```bash
#!/bin/bash
# detect_drift.sh — Daily cron job to catch manual changes

set -e

echo "=== Drift Detection $(date) ==="

# Refresh state from reality (queries provider APIs)
terraform refresh -no-color > /tmp/refresh.log 2>&1

# Run plan to detect differences
terraform plan -detailed-exitcode -no-color > /tmp/plan.log 2>&1
EXIT_CODE=$?

# Exit codes:
#   0 = no changes (state matches code and reality)
#   1 = error
#   2 = changes detected (drift or unmerged code)

if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ No drift detected"
  exit 0
elif [ $EXIT_CODE -eq 2 ]; then
  echo "⚠️  DRIFT DETECTED — manual changes outside Terraform!"
  echo "Plan output:"
  cat /tmp/plan.log
  
  # Alert team (Slack, PagerDuty, etc.)
  curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
    -d "{\"text\": \"🚨 Terraform drift detected in production! Check drift_detection log.\"}"
  
  exit 2
else
  echo "❌ Error running drift detection"
  cat /tmp/plan.log
  exit 1
fi
```

**Decision checkpoint #4 — Responding to detected drift:**

```
Scenario: Daily drift detection alerts that security group rules changed

Drift output:
  ~ aws_security_group.web
    ~ ingress {
      + cidr_blocks = ["0.0.0.0/0"]  ← Someone manually added this rule!
        from_port   = 22
        to_port     = 22
        protocol    = "tcp"
      }

Investigation:
  - Check AWS CloudTrail: Who made the change?
    → engineer@example.com opened SSH to world (for debugging)
  - Was this intentional? Ask engineer
    → "Temporary fix for urgent access issue, forgot to revert"

DECISION TREE:
  Option A: Revert to code (close security hole)
    → Run: terraform apply  # Removes the 0.0.0.0/0 rule
    → Result: Drift eliminated, security restored
    → Post-mortem: Why did engineer bypass Terraform?
  
  Option B: Update code to match reality (if change was valid)
    → Update Terraform config with new rule
    → Commit to Git (with justification in PR)
    → Run: terraform apply (no changes, just syncs state)
    → Result: Drift eliminated, change preserved
  
  Recommended: OPTION A (0.0.0.0/0 on SSH is never valid for production)
```

**State backup and rollback strategy:**

```bash
# Backup state before risky apply
$ cp terraform.tfstate terraform.tfstate.backup-$(date +%Y%m%d-%H%M%S)

# Or use S3 versioning (automatic)
$ aws s3api list-object-versions \
    --bucket my-company-terraform-state \
    --prefix infrastructure/prod/terraform.tfstate

# Rollback to previous state version if apply goes wrong
$ aws s3api get-object \
    --bucket my-company-terraform-state \
    --key infrastructure/prod/terraform.tfstate \
    --version-id <VERSION_ID> \
    terraform.tfstate

$ terraform plan  # Verify rollback worked
```

**Decision checkpoint #5 — State backend selection:**

```
Decision matrix: Which state backend for your team?

┌─────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Backend         │ Local File   │ S3+DynamoDB  │ Terraform    │ Azure Blob   │
│                 │              │              │ Cloud        │ + Lock       │
├─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Cost            │ Free         │ ~$1/month    │ $20+/user/mo │ ~$2/month    │
│ State locking   │ ❌ No        │ ✅ Yes       │ ✅ Yes       │ ✅ Yes       │
│ Versioning      │ Manual       │ ✅ Automatic │ ✅ Automatic │ ✅ Automatic │
│ Encryption      │ ❌ No        │ ✅ Yes       │ ✅ Yes       │ ✅ Yes       │
│ Team collab     │ ❌ Poor      │ ✅ Good      │ ✅ Excellent │ ✅ Good      │
│ Setup time      │ 0 min        │ 10 min       │ 5 min        │ 10 min       │
│ Best for        │ Solo dev     │ Small teams  │ Enterprise   │ Azure users  │
└─────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

RECOMMENDATION BY SCENARIO:
  Solo developer / learning Terraform:
    → Local file (simplicity, no overhead)
  
  Team of 2-10 engineers, AWS-based:
    → S3 + DynamoDB (cost-effective, battle-tested)
  
  Team >10 engineers, need RBAC / policy enforcement:
    → Terraform Cloud (centralized, audit logs, policy as code)
  
  Azure-native environment:
    → Azure Blob Storage + State Lock (native integration)
```

**Exit criteria:** Remote backend configured, state locking enabled, drift detection running, rollback strategy documented. Infrastructure is production-ready and maintainable long-term.

> 💡 **Industry callout #5 — Terragrunt for DRY Terraform configs:** Terragrunt (open-source by Gruntwork, 7k+ stars) wraps Terraform to reduce duplication across environments (dev/staging/prod) — define backend config once, share modules, manage dependencies. Solves "copy-paste Terraform configs that drift over time" problem. Alternative: **Terraform workspaces** (built-in, simpler but limited) and **custom wrapper scripts** (flexible but requires maintenance). Best practice: use Terragrunt for >3 environments with shared modules; raw Terraform for simpler setups.

---

## 2 · Provisioning Docker Containers with Terraform from Zero

You're a platform engineer at a SaaS startup. Your team needs reproducible environments for dev, staging, and prod — identical infrastructure defined as code. The CTO wants to start small: use Terraform to manage Docker containers locally before moving to cloud resources.

**The running example:**
- Provision an Nginx container with Terraform
- Map port 8080 → 80 for local access
- Mount a volume for custom HTML content
- Use variables for container name and port (reusable across environments)
- Output the container's IP address after provisioning

**Constraint:** Must run entirely locally (no cloud dependencies) — Terraform uses the Docker provider to manage containers on your local Docker daemon.

---

## [Phase 1: DEFINE] 3 · The Terraform Workflow at a Glance

Before diving into `.tf` file syntax, here's the full workflow you'll follow. Each numbered step has a corresponding section below.

```
1. Write Terraform configuration files (.tf)                    ← Phase 1: DEFINE
 └─ main.tf: define resources (Docker container)
 └─ variables.tf: parameterize values (container name, port)
 └─ outputs.tf: export values (container IP)

2. Initialize Terraform (terraform init)                        ← Phase 1: DEFINE
 └─ Downloads provider plugins (Docker, AWS, Azure, etc.)
 └─ Creates .terraform/ directory with plugin cache
 └─ Initializes backend for state storage

3. Plan changes (terraform plan)                                ← Phase 2: PLAN
 └─ Compares desired state (code) vs. current state (tfstate + reality)
 └─ Shows preview: "+ create", "~ update", "- destroy"
 └─ NO side effects — safe to run repeatedly

4. Apply changes (terraform apply)                              ← Phase 3: APPLY
 └─ Executes the plan (provisions resources)
 └─ Updates state file (terraform.tfstate)
 └─ Displays outputs

5. Inspect state (terraform show)                               ← Phase 4: VALIDATE
 └─ View current managed resources
 └─ Useful for debugging and auditing

6. Update infrastructure (edit .tf → plan → apply)              ← Phases 2-3 (iterative)
 └─ Modify code (change container image, port, etc.)
 └─ Re-run plan to preview changes
 └─ Apply to update infrastructure

7. Destroy infrastructure (terraform destroy)                   ← Phase 5: MAINTAIN
 └─ Removes all managed resources
 └─ Updates state file to empty
 └─ Useful for ephemeral environments (CI testing, demos)
```

**Notation:**
- **Resource block** — defines infrastructure component. Syntax: `resource "type" "name" { ... }`
- **Provider** — plugin that translates Terraform commands to API calls (Docker, AWS, etc.)
- **State file** — JSON file tracking managed infrastructure (`terraform.tfstate`)
- **Plan** — diff between desired and current state (shown before apply)
- **Drift** — when someone manually changes infrastructure outside Terraform

Sections 4–7 explain each component. Come back to this map when the detail feels overwhelming.

> 💡 **How this maps to the 5-phase workflow (§1.5):** Steps 1-2 are Phase 1 (DEFINE), step 3 is Phase 2 (PLAN), step 4 is Phase 3 (APPLY), step 5 is Phase 4 (VALIDATE), and steps 6-7 are Phase 5 (MAINTAIN). Read §1.5 for the practitioner decision framework at each phase.

---

## [Phase 3: APPLY] 4 · The Math Defines State Reconciliation and Resource Dependencies

> 💡 **Phase context:** This section explains the mathematical foundations that power Phase 2 (PLAN — computing diffs) and Phase 3 (APPLY — executing changes in correct order). Understanding dependency graphs and set operations helps you debug "resource X depends on Y" errors and predict terraform plan output.

### [Phase 3: APPLY] 4.1 · Terraform Computes a Directed Acyclic Graph of Resource Dependencies

Terraform parses your `.tf` files and builds a dependency graph. Resources that don't depend on each other are provisioned in parallel.

Example:
```hcl
resource "docker_network" "private_network" {
  name = "my_network"
}

resource "docker_container" "web" {
  name  = "nginx"
  image = "nginx:latest"
  
  networks_advanced {
    name = docker_network.private_network.name  # Explicit dependency
  }
}
```

**Dependency graph:**
```
docker_network.private_network
    ↓
docker_container.web
```

Terraform ensures the network is created *before* the container. If you try to create them manually in the wrong order, the container creation fails (network doesn't exist yet). Terraform solves this automatically.

**What does the graph mean?** Each node is a resource. An edge from A → B means "B depends on A". Terraform uses topological sort to determine the creation order. Resources at the same level (no dependencies) are created in parallel.

### [Phase 2: PLAN] 4.2 · State Reconciliation Uses Set Difference to Compute Changes

> 💡 **Phase context:** This is the mathematical core of `terraform plan` — how Terraform computes the diff between what you want (code) and what exists (state + reality).

Define:
- $D$ = desired state (resources in `.tf` files)
- $C$ = current state (resources in `terraform.tfstate`)
- $R$ = reality (actual infrastructure queried from provider APIs)

Terraform computes:

$$\text{Create} = D \setminus C \quad \text{(resources in desired but not current)}$$

$$\text{Destroy} = C \setminus D \quad \text{(resources in current but not desired)}$$

$$\text{Update} = \{r \in D \cap C \mid r_D \neq r_C\} \quad \text{(resources with config changes)}$$

**Example:** You have 2 containers in state (`web`, `db`). You remove `db` from your `.tf` files and add `cache`. Terraform computes:
- Create: `cache` (in $D$, not in $C$)
- Destroy: `db` (in $C$, not in $D$)
- Update: none (if `web` config unchanged)

**Why the state file exists:** Terraform can't rely solely on querying provider APIs (reality check) because:
1. Some resources don't have read APIs (e.g., one-time tokens)
2. Querying thousands of resources on every plan is slow
3. State file tracks metadata (e.g., dependencies) not stored in provider APIs

The state file is the **source of truth** for what Terraform manages. If you delete it, Terraform thinks all resources are new (tries to create duplicates, fails on name conflicts).

### [Phase 5: MAINTAIN] 4.3 · Drift Detection Compares State File Against Reality

> 💡 **Phase context:** This section explains how Phase 5 (MAINTAIN) detects manual changes outside Terraform. Production teams run drift detection daily to catch unauthorized infrastructure modifications.

> 💡 **Intuition first:** **Terraform's state file prevents drift by acting as a single source of truth for your infrastructure**. Without state, Terraform would have to query every provider API ("does this container exist? what port is it using?") on every run — slow and error-prone. The state file caches this information and adds a **locking mechanism** (via remote backends like S3) to prevent two engineers from applying conflicting changes simultaneously. Think of state as **Terraform's memory**: it remembers what it built, compares it to what you want (your `.tf` files), and calculates the minimal diff. When someone makes manual changes outside Terraform, state detects **drift** — the gap between what Terraform thinks exists (state file) and what actually exists (reality).

**Drift** happens when someone manually changes infrastructure (AWS console, Docker CLI, etc.) *outside* Terraform. Example:
1. Terraform creates a container with port 8080
2. You manually change the port to 9090 using Docker CLI
3. State file still says port 8080
4. Reality (Docker daemon) says port 9090

When you run `terraform plan`, Terraform:
1. Reads state file: "port should be 8080"
2. Queries Docker API: "port is currently 9090"
3. Detects drift: "reality doesn't match state"
4. Plans to revert: "~ update port 9090 → 8080"

**Best practice:** Never manually change Terraform-managed resources. All changes should go through code → plan → apply. Use `terraform refresh` to sync state with reality if needed (but this doesn't fix drift — it just updates the state file to match reality).

---

## [Phase 4: VALIDATE] 5 · What Can Go Wrong — Production Pitfalls and How to Avoid Them

> 💡 **Phase context:** This section covers common failure modes you'll encounter in Phase 3 (APPLY), Phase 4 (VALIDATE), and Phase 5 (MAINTAIN). Each pitfall includes detection steps and remediation strategies.

### [Phase 5: MAINTAIN] 5.1 · State File Corruption

**Symptom:** `terraform apply` fails with "resource already exists" or tries to recreate everything.

**Cause:** State file got corrupted, deleted, or out of sync with reality.

**Fix:**
1. **Never edit state file manually** — use `terraform state` commands
2. **Use remote state backend** (S3, Azure Blob, Terraform Cloud) — enables locking and prevents concurrent modifications
3. **Backup state file** — commit to Git (if not sensitive) or use automated backups

**Prevention:**
```hcl
# Use remote backend (S3 example)
terraform {
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-west-2"
  }
}
```

### [Phase 1: DEFINE] 5.2 · Provider Version Conflicts

**Symptom:** `terraform init` fails with "no matching version" or code works on your machine but fails in CI.

**Cause:** Provider version not pinned — Terraform downloads different versions on different machines.

**Fix:** Always pin provider versions with `~>` constraint:
```hcl
terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"  # Allow 3.x but not 4.0
    }
  }
  required_version = ">= 1.5"  # Require Terraform 1.5+
}
```

**Best practice:** Commit `.terraform.lock.hcl` to Git — locks exact provider versions across team.

### [Phase 5: MAINTAIN] 5.3 · Drift Detection Ignored

**Symptom:** Infrastructure behaves differently than code suggests — manual changes not tracked.

**Cause:** Someone modified resources outside Terraform (AWS console, kubectl, Docker CLI).

**Fix:** Run `terraform plan` regularly to detect drift. If drift detected:
1. **Option A:** Let Terraform revert changes (`terraform apply` reverts to code)
2. **Option B:** Update code to match reality (`terraform refresh` + commit changes)

**Prevention:** Enforce policy — *all infrastructure changes go through Terraform*. Use CI checks to prevent manual changes:
```bash
# In CI pipeline
terraform plan -detailed-exitcode
# Exit code 2 = changes detected (drift or unmerged code)
```

### [Phase 5: MAINTAIN] 5.4 · Secrets in State File

**Symptom:** `terraform.tfstate` contains database passwords, API keys in plaintext.

**Cause:** State file stores *all* resource attributes, including sensitive values.

**Fix:**
1. **Never commit state file to public repos**
2. **Use remote backend with encryption** (S3 with encryption, Azure with RBAC)
3. **Mark outputs as sensitive**:
```hcl
output "db_password" {
  value     = aws_db_instance.main.password
  sensitive = true  # Won't display in console output
}
```

**Best practice:** Use separate state files per environment (dev, prod) — limits blast radius if state leaks.

### [Phase 3: APPLY] 5.5 · Terraform Destroy in Production

**Symptom:** Someone runs `terraform destroy` in production by mistake — deletes all infrastructure.

**Cause:** No safeguards against destructive commands.

**Fix:**
1. **Use lifecycle rules** to prevent accidental deletion:
```hcl
resource "aws_db_instance" "main" {
  # ... config ...
  
  lifecycle {
    prevent_destroy = true  # Terraform will refuse to destroy this
  }
}
```

2. **Require approval for production applies** — use Terraform Cloud or GitHub Actions with manual approval step

3. **Separate workspaces** — dev, staging, prod use different state files

---

## 6 · Progress Check — Test Your Understanding

Before moving to Ch.7, verify you can:

1. **Predict plan output:** Given this diff:
   ```diff
   resource "docker_container" "app" {
     name  = "my_app"
   - image = "nginx:1.24"
   + image = "nginx:1.25"
     ports {
       internal = 80
       external = 8080
     }
   }
   ```
   What does `terraform plan` show?  
   **Answer:** `~ update in-place` (container image changed) — Terraform will recreate the container (can't update image on running container).

2. **Debug state mismatch:** You run `terraform plan` and see:
   ```
   ~ docker_container.app
     ~ ports[0].external: 8080 → 9090
   ```
   But your code shows `external = 8080`. What happened?  
   **Answer:** Drift — someone manually changed the port outside Terraform. Run `terraform apply` to revert, or update code to match reality.

3. **Identify dependency order:** Given these resources:
   ```hcl
   resource "docker_container" "db" { ... }
   resource "docker_network" "net" { ... }
   resource "docker_container" "app" {
     networks_advanced {
       name = docker_network.net.name
     }
     depends_on = [docker_container.db]
   }
   ```
   What's the creation order?  
   **Answer:** `docker_network.net` → `docker_container.db` → `docker_container.app` (network first, then db and app in parallel, but app explicitly waits for db).

**Common mistakes:**
- ❌ Running `terraform apply` without reviewing plan first
- ❌ Manually editing infrastructure then forgetting to update code
- ❌ Deleting state file to "start fresh" (loses track of existing resources)
- ❌ Not using version control for `.tf` files (defeats purpose of IaC)

---

## 7 · Bridge to Ch.7 — Networking Makes Services Discoverable

You've mastered Infrastructure as Code — you can provision Docker containers, networks, and volumes with Terraform. Changes are version-controlled, peer-reviewed, and reproducible. **But your services can't talk to each other yet.**

**What we've built:**
- ✅ Provision infrastructure declaratively (no manual steps)
- ✅ Track changes in Git (rollback, review, audit trail)
- ✅ Detect drift when infrastructure diverges from code

**What's still missing:**
Your containers are isolated. You can spin up 3 web servers and 1 database, but:
- How do web servers *discover* the database IP?
- How do you *load balance* traffic across the 3 web servers?
- How do you expose services to the internet while keeping internal services private?

**Ch.7 — Networking & Load Balancing** covers:
- **Service discovery**: How containers find each other (DNS, service names)
- **Load balancing**: Distribute traffic across replicas (Nginx, Traefik, cloud LBs)
- **Network segmentation**: Public vs. private networks, firewall rules
- **Reverse proxies**: SSL termination, routing by hostname/path

**The bridge:** Terraform provisions the infrastructure. Networking makes it functional.

**Before moving on:** Make sure you can:
1. Write `.tf` files to provision resources
2. Use `terraform plan` to preview changes
3. Apply changes and inspect state
4. Detect and resolve drift

You're ready when infrastructure provisioning is a *code commit*, not a manual checklist.
