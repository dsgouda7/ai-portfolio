#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
echo "Checking Docker installation..."
DOCKER_INSTALLED=false

if command -v docker &>/dev/null; then
    docker_version=$(docker --version 2>&1)
    DOCKER_INSTALLED=true
    echo "Found Docker: $docker_version"
else
    echo "Docker is not currently installed."
fi

# Install Docker if missing
if [ "$DOCKER_INSTALLED" = false ]; then
    echo "Attempting to install Docker..."

    case "$OS" in
        debian)
            echo "Installing Docker Engine via apt..."
            sudo apt-get update
            sudo apt-get install -y ca-certificates curl gnupg lsb-release

            sudo mkdir -p /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

            echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

            sudo apt-get update
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            sudo usermod -aG docker "$USER"

            echo "Docker installed successfully."
            echo "Please log out and log back in, then re-run this script."
            exit 0
            ;;

        macos)
            if command -v brew &>/dev/null; then
                echo "Installing Docker Desktop via Homebrew..."
                brew install --cask docker
                echo "Docker Desktop installed successfully."
                echo "Please open Docker from your Applications folder, wait for it to start, then re-run this script."
                exit 0
            else
                echo "Error: Homebrew not found. Install Docker Desktop manually:" >&2
                echo "https://www.docker.com/products/docker-desktop/"
                exit 1
            fi
            ;;

        *)
            echo "Error: Automatic installation not supported for this OS. Install Docker manually:" >&2
            echo "https://docs.docker.com/engine/install/"
            exit 1
            ;;
    esac
fi

# Verify Docker Daemon is running
if [ "$DOCKER_INSTALLED" = true ]; then
    echo "Checking Docker daemon status..."
    if ! docker info &>/dev/null; then
        echo "Error: Docker daemon is not running. Please start Docker and try again." >&2
        exit 1
    fi
    echo "Docker daemon is active."
fi

# Verify Project Integrity
echo "Verifying required project files..."
required_files=("Dockerfile" "requirements.txt" "model_manager.py" "controllers.py")
missing_files=0

for file in "${required_files[@]}"; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        echo "  OK: $file"
    else
        echo "  Missing: $file" >&2
        missing_files=$((missing_files + 1))
    fi
done

if [ "$missing_files" -gt 0 ]; then
    echo "Error: Workspace is incomplete. Resolve missing files before proceeding." >&2
    exit 1
fi

# Check Optional Python Environment (Local Dev)
echo "Checking for local Python installation..."
PYTHON_FOUND=false

for python_cmd in python3 python; do
    if command -v "$python_cmd" &>/dev/null; then
        py_version=$("$python_cmd" --version 2>&1)
        echo "Found local Python: $py_version"
        PYTHON_FOUND=true
        break
    fi
done

if [ "$PYTHON_FOUND" = false ]; then
    echo "Python not found. (This is fine if you only plan to run within Docker)"
fi

# Success Output
echo -e "\nEnvironment verification complete."
echo "To start the application, execute: ./run.sh"
