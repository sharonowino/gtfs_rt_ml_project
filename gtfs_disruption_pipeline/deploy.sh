#!/bin/bash
# GTFS Disruption Pipeline - Deployment Script
# =========================================

set -e

echo "=== GTFS Disruption Pipeline Deployment ==="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default values
IMAGE_NAME="gtfs-disruption-api"
REGISTRY=""
DEPLOYMENT_ENV="production"
MODEL_PATH="/app/models/best_model.pkl"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=true
            shift
            ;;
        --deploy)
            DEPLOY=true
            shift
            ;;
        --env)
            DEPLOYMENT_ENV="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build Docker image
build() {
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME:latest .
    success "Docker image built"
    
    if [ -n "$REGISTRY" ]; then
        docker tag $IMAGE_NAME:latest $REGISTRY/$IMAGE_NAME:latest
        docker push $REGISTRY/$IMAGE_NAME:latest
        success "Image pushed to registry"
    fi
}

# Deploy with docker-compose
deploy() {
    echo "Deploying with docker-compose..."
    
    # Export model path
    export MODEL_PATH
    
    # Start services
    docker-compose up -d
    success "Services started"
    
    # Wait for API
    echo "Waiting for API..."
    sleep 10
    
    # Health check
    HEALTH=$(curl -s http://localhost:8000/health/ready)
    if echo "$HEALTH" | grep -q "ready"; then
        success "API is ready"
    else
        warn "API health check returned: $HEALTH"
    fi
    
    # Show status
    docker-compose ps
}

# Main
main() {
    if [ "$BUILD" = true ]; then
        build
    fi
    
    if [ "$DEPLOY" = true ]; then
        deploy
    fi
    
    if [ -z "$BUILD" ] && [ -z "$DEPLOY" ]; then
        echo "Usage: $0 [--build] [--deploy] [--env production] [--registry my-registry] [--model /path/to/model]"
        echo ""
        echo "Options:"
        echo "  --build          Build Docker image"
        echo "  --deploy        Deploy with docker-compose"
        echo "  --env           Deployment environment (default: production)"
        echo "  --registry     Docker registry URL"
        echo "  --model        Path to model file"
    fi
}

main