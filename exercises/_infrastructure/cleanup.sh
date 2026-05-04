#!/usr/bin/env bash
# Cleanup script: Stop containers, remove images, and free disk space
#
# Usage:
#   ./cleanup.sh --exercise "01-ml/01-regression" --all
#   ./cleanup.sh --everything  # Nuclear option: clean ALL Docker resources

set -e

EXERCISE=""
ALL=false
EVERYTHING=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exercise)
            EXERCISE="$2"
            shift 2
            ;;
        --all)
            ALL=true
            shift
            ;;
        --everything)
            EVERYTHING=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "\n🧹 Docker Cleanup Script"
echo "═══════════════════════════════════════"

if [[ "$EVERYTHING" == true ]]; then
    echo -e "\n⚠️  WARNING: Nuclear cleanup - will remove ALL Docker resources!"
    echo "   This affects ALL projects on your machine, not just this repo."
    read -p "Type 'yes' to confirm: " confirm
    
    if [[ "$confirm" != "yes" ]]; then
        echo -e "\n❌ Cancelled"
        exit 0
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "\n📋 DRY RUN - Would execute:"
        echo "  docker-compose -f $INFRA_DIR/docker-compose-generated.yml down"
        echo "  docker container prune -f"
        echo "  docker image prune -a -f"
        echo "  docker volume prune -f"
        echo "  docker network prune -f"
        exit 0
    fi
    
    echo -e "\n🛑 Stopping all containers..."
    [[ -f "$INFRA_DIR/docker-compose-generated.yml" ]] && docker-compose -f "$INFRA_DIR/docker-compose-generated.yml" down
    
    echo -e "\n🗑️  Removing all stopped containers..."
    docker container prune -f
    
    echo -e "\n🗑️  Removing all unused images..."
    docker image prune -a -f
    
    echo -e "\n🗑️  Removing all unused volumes..."
    docker volume prune -f
    
    echo -e "\n🗑️  Removing all unused networks..."
    docker network prune -f
    
    echo -e "\n✅ Nuclear cleanup complete!"
    docker system df
    exit 0
fi

if [[ -z "$EXERCISE" && "$EVERYTHING" != true ]]; then
    echo "Error: Must specify --exercise or use --everything"
    echo -e "\nUsage:"
    echo "  ./cleanup.sh --exercise '01-ml/01-regression' --all"
    echo "  ./cleanup.sh --everything  # Clean ALL Docker resources"
    exit 1
fi

PROJECT_NAME="${EXERCISE//\//-}-api"

echo -e "\n📦 Exercise: $EXERCISE"
echo "   Project: $PROJECT_NAME"

if [[ "$DRY_RUN" == true ]]; then
    echo -e "\n📋 DRY RUN - Would execute:"
    echo "  docker-compose -f $INFRA_DIR/docker-compose-generated.yml down"
    if [[ "$ALL" == true ]]; then
        echo "  docker rmi $PROJECT_NAME -f"
        echo "  docker rmi prom/prometheus:latest -f"
        echo "  docker rmi grafana/grafana:latest -f"
    fi
    exit 0
fi

# Stop containers
echo -e "\n🛑 Stopping containers..."
if [[ -f "$INFRA_DIR/docker-compose-generated.yml" ]]; then
    docker-compose -f "$INFRA_DIR/docker-compose-generated.yml" down
    echo "   ✓ Containers stopped"
else
    echo "   ⚠️  No docker-compose-generated.yml found"
fi

# Remove images if --all flag
if [[ "$ALL" == true ]]; then
    echo -e "\n🗑️  Removing Docker images..."
    
    if docker images -q $PROJECT_NAME 2>/dev/null; then
        docker rmi $PROJECT_NAME -f
        echo "   ✓ Removed $PROJECT_NAME"
    else
        echo "   ⚠️  Image $PROJECT_NAME not found"
    fi
    
    echo -e "\n⚠️  Also remove shared monitoring images (prometheus, grafana)?"
    read -p "Remove shared images? (y/N): " remove_shared
    
    if [[ "$remove_shared" == "y" || "$remove_shared" == "Y" ]]; then
        docker rmi prom/prometheus:latest -f 2>/dev/null || true
        docker rmi grafana/grafana:latest -f 2>/dev/null || true
        echo "   ✓ Removed shared images"
    fi
fi

echo -e "\n✅ Cleanup complete!"
echo -e "\n📊 Current Docker disk usage:"
docker system df
