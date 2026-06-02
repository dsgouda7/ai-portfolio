#!/usr/bin/env bash
# Voice Assistant - Setup Script (Bash)
# Installs Docker if necessary and sets up the environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Color output functions
step() { echo; echo "▶ $*"; }
ok()   { echo "  ✓ $*"; }
warn() { echo "  ⚠ $*"; }
fail() { echo "  ✗ $*"; exit 1; }

echo
echo "╔════════════════════════════════════════╗"
echo "║   Voice Assistant - Setup (Unix/Mac)   ║"
echo "╚════════════════════════════════════════╝"
echo

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v apt-get &>/dev/null; then
        OS="debian"
    elif command -v dnf &>/dev/null; then
        OS="fedora"
    elif command -v pacman &>/dev/null; then
        OS="arch"
    else
        OS="linux"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
fi

echo "Detected OS: $OS"

# Check if Docker is installed
step "Checking Docker installation..."
DOCKER_INSTALLED=false

if command -v docker &>/dev/null; then
    docker_version=$(docker --version 2>&1)
    if [ $? -eq 0 ]; then
        DOCKER_INSTALLED=true
        ok "Docker found: $docker_version"
    fi
else
    warn "Docker not found"
fi

# Install Docker if not found
if [ "$DOCKER_INSTALLED" = false ]; then
    step "Installing Docker..."

    case "$OS" in
        debian)
            warn "Installing Docker on Debian/Ubuntu requires sudo..."
            sudo apt-get update
            sudo apt-get install -y \
                ca-certificates \
                curl \
                gnupg \
                lsb-release

            # Add Docker's official GPG key
            sudo mkdir -p /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

            # Set up repository
            echo \
              "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
              $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

            # Install Docker Engine
            sudo apt-get update
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

            # Add current user to docker group
            sudo usermod -aG docker "$USER"

            ok "Docker installed successfully"
            warn "Please log out and log back in for group changes to take effect"
            warn "Then run this setup script again"
            exit 0
            ;;

        macos)
            if command -v brew &>/dev/null; then
                warn "Installing Docker Desktop via Homebrew..."
                brew install --cask docker
                ok "Docker Desktop installed"
                warn "Please start Docker Desktop from Applications and wait for it to initialize"
                warn "Then run this setup script again"
                exit 0
            else
                warn "Homebrew not found"
                echo
                echo "  Please install Docker Desktop manually:"
                echo "  1. Visit: https://www.docker.com/products/docker-desktop/"
                echo "  2. Download Docker Desktop for Mac"
                echo "  3. Install and start Docker Desktop"
                echo "  4. Run this setup script again"
                echo
                exit 1
            fi
            ;;

        *)
            warn "Automatic installation not available for this OS"
            echo
            echo "  Please install Docker manually:"
            echo "  1. Visit: https://docs.docker.com/engine/install/"
            echo "  2. Follow instructions for your distribution"
            echo "  3. Run this setup script again"
            echo
            exit 1
            ;;
    esac
fi

# Check if Docker daemon is running
if [ "$DOCKER_INSTALLED" = true ]; then
    step "Checking Docker daemon..."
    if docker info &>/dev/null; then
        ok "Docker daemon is running"
    else
        fail "Docker daemon is not running. Please start Docker and try again."
    fi
fi

# Verify required files exist
step "Verifying project files..."
required_files=("Dockerfile" "requirements.txt" "model_manager.py" "controllers.py")
all_files_exist=true

for file in "${required_files[@]}"; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        ok "$file exists"
    else
        fail "$file not found"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    fail "Missing required files"
fi

# Check for Python (optional - for local testing without Docker)
step "Checking Python installation (optional)..."
PYTHON_FOUND=false

for python_cmd in python3 python; do
    if command -v "$python_cmd" &>/dev/null; then
        py_version=$("$python_cmd" --version 2>&1)
        ok "Python found: $py_version"
        PYTHON_FOUND=true
        break
    fi
done

if [ "$PYTHON_FOUND" = false ]; then
    warn "Python not found (only needed for local testing without Docker)"
fi

# Summary
echo
echo "╔════════════════════════════════════════╗"
echo "║      Setup Complete! ✓                 ║"
echo "╚════════════════════════════════════════╝"
echo
echo "Next steps:"
echo "  1. Run: ./run.sh"
echo "  2. Wait for models to download (first run only)"
echo "  3. Access the app at: http://localhost:5000"
echo
