#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

echo "Checking Docker installation..."
DOCKER_INSTALLED=false

if command -v docker &>/dev/null; then
    docker_version=$(docker --version 2>&1)
    if [ $? -eq 0 ]; then
        DOCKER_INSTALLED=true
        echo "Found Docker: $docker_version"
    fi
else
    echo "Docker is not currently installed."
fi

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
            echo "Docker installed."
            echo "Please log out and log back in, then re-run this script."
            exit 0
            ;;

        macos)
            if command -v brew &>/dev/null; then
                echo "Installing Docker Desktop via Homebrew..."
                brew install --cask docker
                echo "Docker Desktop installed."
                echo "Please launch Docker Desktop and wait for initialization."
                exit 0
            else
                echo "Error: Homebrew not found. Install Docker Desktop manually:"
                echo "https://www.docker.com/products/docker-desktop/"
                exit 1
            fi
            ;;

        *)
            echo "Error: Automatic installation not available for this OS."
            echo "Please install Docker manually: https://docs.docker.com/engine/install/"
            exit 1
            ;;
    esac
fi

if [ "$DOCKER_INSTALLED" = true ]; then
    echo "Checking Docker daemon..."
    if docker info &>/dev/null; then
        echo "Docker daemon is active."
    else
        echo "Docker daemon is not running. Attempting to start Docker..."

        case "$OS" in
            macos)
                if [ -d "/Applications/Docker.app" ]; then
                    echo "Starting Docker Desktop..."
                    open -a Docker

                    echo "Waiting for Docker to start..."
                    max_attempts=30
                    attempt=0

                    while [ $attempt -lt $max_attempts ]; do
                        sleep 2
                        if docker info &>/dev/null; then
                            echo "Docker daemon started successfully."
                            break
                        fi
                        attempt=$((attempt + 1))
                        echo -n "."
                    done

                    if [ $attempt -eq $max_attempts ]; then
                        echo ""
                        echo "Error: Docker daemon failed to start within 60 seconds. Please start Docker manually."
                        exit 1
                    fi
                else
                    echo "Error: Docker Desktop not found. Please start Docker manually."
                    exit 1
                fi
                ;;

            debian|fedora|arch|linux)
                echo "Attempting to start Docker service..."
                if command -v systemctl &>/dev/null; then
                    sudo systemctl start docker
                    sleep 3

                    if docker info &>/dev/null; then
                        echo "Docker daemon started successfully."
                    else
                        echo "Error: Failed to start Docker daemon. Please start it manually: sudo systemctl start docker"
                        exit 1
                    fi
                else
                    echo "Error: Docker daemon is not running. Please start it manually."
                    exit 1
                fi
                ;;

            *)
                echo "Error: Docker daemon is not running. Please start Docker manually."
                exit 1
                ;;
        esac
    fi
fi

echo "Verifying required project files..."
required_files=("Dockerfile" "requirements.txt" "video_processor.py" "audio_processor.py" "app.py")
missing_files=0

for file in "${required_files[@]}"; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        echo "  OK: $file"
    else
        echo "  Missing: $file"
        missing_files=$((missing_files + 1))
    fi
done

if [ $missing_files -gt 0 ]; then
    echo "Error: Workspace is incomplete. Resolve missing files before proceeding."
    exit 1
fi

echo "Checking for local Python installation..."
for python_cmd in python3 python; do
    if command -v "$python_cmd" &>/dev/null; then
        py_version=$("$python_cmd" --version 2>&1)
        echo "Found Python: $py_version (optional for local testing)"
        break
    fi
done

echo -e "\nSetup complete. Run './run.sh' to start the application."
