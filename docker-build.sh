#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

ERROR_COUNT=0
WARN_COUNT=0

handle_error() {
    ERROR_COUNT=$((ERROR_COUNT + 1))
    local msg="$1"
    log_error "$msg"
}

handle_warning() {
    WARN_COUNT=$((WARN_COUNT + 1))
    local msg="$1"
    log_warn "$msg"
}

cleanup_deprecated_env_keys() {
    local env_file="$1"
    [[ -f "${env_file}" ]] || return 0
    local deprecated_keys=(
        "OLLAMA_API_KEY"
        "OLLAMA_MODEL"
        "OLLAMA_HOST"
        "OLLAMA_ENABLED"
        "OLLAMA_TEMPERATURE"
        "OLLAMA_MAX_TOKENS"
        "LLM_PROVIDER"
    )
    for key in "${deprecated_keys[@]}"; do
        sed -i "/^${key}=/d" "${env_file}"
    done
}

CLEANUP_ON_EXIT=true
CLOUDFLARED_PID=""
DOCKER_RUNNING=true

cleanup() {
    if [[ "$CLEANUP_ON_EXIT" == "true" ]]; then
        echo ""
        log_info "Cleaning up..."
        
        if [[ -n "$CLOUDFLARED_PID" ]] && kill -0 "$CLOUDFLARED_PID" 2>/dev/null; then
            kill "$CLOUDFLARED_PID" 2>/dev/null || true
            log_info "Cloudflare Tunnel stopped"
        fi
        
        log_info "Container continues running in background"
    fi
}

trap cleanup SIGINT SIGTERM EXIT

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Reco-Trading Docker Builder v5.0${NC}"
echo -e "${CYAN}  Fixed & Optimized${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

log_info "Checking prerequisites..."

check_command() {
    command -v "$1" >/dev/null 2>&1
}

log_info "Detecting system resources..."

CPU_CORES=2
if [[ -f /proc/cpuinfo ]]; then
    CPU_CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "2")
elif command -v sysctl >/dev/null 2>&1; then
    CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "2")
fi

TOTAL_MEM_KB=0
RAM_GB=2
if [[ -f /proc/meminfo ]]; then
    TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    RAM_GB=$((TOTAL_MEM_KB / 1024 / 1024))
    if [[ $RAM_GB -eq 0 ]]; then
        RAM_GB=2
    fi
elif command -v sysctl >/dev/null 2>&1; then
    RAM_GB=$(sysctl -n hw.memsize 2>/dev/null || echo "2048")
    RAM_GB=$((RAM_GB / 1024 / 1024 / 1024))
fi

DISK_GB=10
if df / >/dev/null 2>&1; then
    DISK_GB=$(df -BG / 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
fi

echo "  CPU Cores: ${CPU_CORES}"
echo "  RAM: ${RAM_GB} GB"
echo "  Disk: ${DISK_GB} GB"

if [[ $RAM_GB -lt 1 ]]; then
    handle_error "Insufficient memory (${RAM_GB}GB). Minimum 1GB required."
    exit 1
fi

if [[ $RAM_GB -ge 16 ]] && [[ $CPU_CORES -ge 4 ]]; then
    RESOURCE_PROFILE="high"
    WORKERS=4
    ML_WORKERS=4
elif [[ $RAM_GB -ge 8 ]] && [[ $CPU_CORES -ge 2 ]]; then
    RESOURCE_PROFILE="medium"
    WORKERS=2
    ML_WORKERS=2
else
    RESOURCE_PROFILE="low"
    WORKERS=1
    ML_WORKERS=1
fi

log_info "Profile: ${RESOURCE_PROFILE} (Workers: ${WORKERS}, ML: ${ML_WORKERS})"

log_info "Checking Docker..."

if ! check_command docker; then
    log_error "Docker not found"
    echo ""
    echo "Install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

log_success "Docker installed"

if ! docker info >/dev/null 2>&1; then
    log_error "Docker daemon not running"
    echo ""
    echo "Start Docker with:"
    echo "  sudo systemctl start docker  # Linux"
    echo "  # Or start Docker Desktop on Windows/Mac"
    exit 1
fi

log_success "Docker daemon active"

if ! docker compose version >/dev/null 2>&1 && ! docker-compose --version >/dev/null 2>&1; then
    log_warn "docker compose not available"
fi

log_info "Checking configuration..."

if [[ ! -f ".env" ]]; then
    log_warn ".env not found, generating template..."
    cat > .env <<'EOF'
# Reco-Trading Configuration
# Generated automatically - UPDATE THESE VALUES!

# Exchange API (REQUIRED)
BINANCE_API_KEY=YOUR_API_KEY_HERE
BINANCE_API_SECRET=YOUR_API_SECRET_HERE
BINANCE_TESTNET=true
CONFIRM_MAINNET=false

# Environment
ENVIRONMENT=testnet
RUNTIME_PROFILE=paper

# Database (PostgreSQL recommended for production)
POSTGRES_DSN=postgresql+asyncpg://trading:trading123@postgres:5432/reco_trading_prod
# Or use SQLite for simple setup:
# DATABASE_URL=sqlite+aiosqlite:///./data/reco_trading.db

# Redis (optional, for caching)
REDIS_URL=redis://redis:6379/0

# LLM/AI Features (optional)
LLM_MODE=base
LLM_REMOTE_ENDPOINT=
LLM_REMOTE_MODEL=
LLM_REMOTE_API_KEY=

# Dashboard Security
DASHBOARD_AUTH_ENABLED=true
DASHBOARD_AUTH_MODE=token
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=admin
DASHBOARD_API_TOKEN=CHANGE_THIS_TOKEN_TO_SOMETHING_SECURE

# Feature Flags
ENABLE_AUTO_IMPROVER=true
ENABLE_ML_ENGINE=true
ENABLE_CONTINUAL_LEARNING=true
ENABLE_META_LEARNING=true
ENABLE_DRIFT_DETECTION=true
ENABLE_ONCHAIN_ANALYSIS=true
ENABLE_EVOLUTION=true
ENABLE_TFT=true
ENABLE_NBEATS=true
ENABLE_ADVANCED_META_LEARNING=true
ENABLE_REINFORCEMENT_LEARNING=true
EOF
    log_warn "Please update .env with your API keys before running!"
    echo ""
    echo "Required: BINANCE_API_KEY and BINANCE_API_SECRET"
    echo ""
    read -p "Press Enter to continue after editing .env..." 
fi

cleanup_deprecated_env_keys ".env"

set -a
source .env 2>/dev/null || true
set +a

if [[ -z "${BINANCE_API_KEY:-}" ]] || [[ "$BINANCE_API_KEY" == "YOUR_API_KEY_HERE" ]] || [[ "$BINANCE_API_KEY" == "CAMBIAR_POR_TU_API_KEY" ]]; then
    log_error "BINANCE_API_KEY not configured in .env"
    echo ""
    echo "Please set BINANCE_API_KEY in .env file"
    exit 1
fi

if [[ -z "${BINANCE_API_SECRET:-}" ]] || [[ "$BINANCE_API_SECRET" == "YOUR_API_SECRET_HERE" ]] || [[ "$BINANCE_API_SECRET" == "CAMBIAR_POR_TU_API_SECRET" ]]; then
    log_error "BINANCE_API_SECRET not configured in .env"
    echo ""
    echo "Please set BINANCE_API_SECRET in .env file"
    exit 1
fi

log_success ".env configured"

IMAGE_EXISTS=false
if docker image ls reco-trading:latest -q 2>/dev/null | grep -q .; then
    IMAGE_EXISTS=true
fi

CONTAINER_EXISTS=false
if docker ps -a --format '{{.Names}}' | grep -q "^reco-trading$"; then
    CONTAINER_EXISTS=true
fi

if [[ "$IMAGE_EXISTS" == "true" ]]; then
    log_warn "Docker image 'reco-trading:latest' already exists"
    echo ""
    echo "Options:"
    echo "  1) Use existing image and create container"
    echo "  2) Delete and rebuild from scratch"
    echo "  3) Update container only (recreate)"
    echo ""
    read -p "Choose option (1/2/3): " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            log_info "Using existing image"
            USE_EXISTING_IMAGE=true
            ;;
        2)
            log_info "Removing old image..."
            docker rmi reco-trading:latest 2>/dev/null || true
            USE_EXISTING_IMAGE=false
            ;;
        3)
            log_info "Updating container..."
            if [[ "$CONTAINER_EXISTS" == "true" ]]; then
                docker stop reco-trading 2>/dev/null || true
                docker rm reco-trading 2>/dev/null || true
            fi
            USE_EXISTING_IMAGE=true
            ;;
        *)
            log_info "Using existing image by default"
            USE_EXISTING_IMAGE=true
            ;;
    esac
elif [[ "$CONTAINER_EXISTS" == "true" ]]; then
    log_warn "Previous container found"
    echo ""
    echo "Options:"
    echo "  1) Stop and remove old container"
    echo "  2) Keep container and exit"
    echo "  3) Update image only"
    echo ""
    read -p "Choose option (1/2/3): " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            log_info "Stopping old container..."
            docker stop reco-trading 2>/dev/null || true
            docker rm reco-trading 2>/dev/null || true
            log_success "Old container removed"
            USE_EXISTING_IMAGE=false
            ;;
        2)
            log_info "Keeping current container"
            CLEANUP_ON_EXIT=false
            docker start reco-trading 2>/dev/null || true
            log_success "Container started"
            exit 0
            ;;
        3)
            log_info "Updating image only..."
            docker stop reco-trading 2>/dev/null || true
            docker rm reco-trading 2>/dev/null || true
            USE_EXISTING_IMAGE=false
            ;;
        *)
            log_info "Using default option"
            USE_EXISTING_IMAGE=true
            ;;
    esac
else
    USE_EXISTING_IMAGE=false
fi

echo ""
log_info "Building Docker image..."

BINANCE_TESTNET="${BINANCE_TESTNET:-true}"
ENVIRONMENT="${ENVIRONMENT:-testnet}"
RUNTIME_PROFILE="${RUNTIME_PROFILE:-paper}"
DATABASE_URL="${DATABASE_URL:-sqlite:///./data/reco_trading.db}"

echo "  Profile: ${RESOURCE_PROFILE}"
echo "  Environment: ${ENVIRONMENT}"
echo "  Testnet: ${BINANCE_TESTNET}"

if [[ "${USE_EXISTING_IMAGE:-false}" == "true" ]] && [[ "$IMAGE_EXISTS" == "true" ]]; then
    log_info "Using existing Docker image..."
    
    if docker image ls reco-trading:latest -q 2>/dev/null | grep -q .; then
        log_success "Existing image verified"
    else
        log_warn "Image no longer exists, building new..."
        USE_EXISTING_IMAGE=false
    fi
fi

if [[ "${USE_EXISTING_IMAGE:-false}" != "true" ]]; then
    BUILD_FAILED=false
    docker build \
        --build-arg RESOURCE_PROFILE="${RESOURCE_PROFILE}" \
        --build-arg WORKERS="${WORKERS}" \
        --build-arg ML_WORKERS="${ML_WORKERS}" \
        --build-arg BINANCE_API_KEY="${BINANCE_API_KEY}" \
        --build-arg BINANCE_API_SECRET="${BINANCE_API_SECRET}" \
        --build-arg BINANCE_TESTNET="${BINANCE_TESTNET}" \
        --build-arg ENVIRONMENT="${ENVIRONMENT}" \
        --build-arg RUNTIME_PROFILE="${RUNTIME_PROFILE}" \
        --build-arg DATABASE_URL="${DATABASE_URL}" \
        --build-arg POSTGRES_DSN="${POSTGRES_DSN:-}" \
        --build-arg MYSQL_DSN="${MYSQL_DSN:-}" \
        --build-arg REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}" \
        -t reco-trading:latest . 2>&1 || BUILD_FAILED=true

    if [[ "$BUILD_FAILED" == "true" ]]; then
        log_error "Failed to build Docker image"
        echo ""
        echo "Check the errors above and try:"
        echo "  - Verify Docker has enough resources"
        echo "  - Check internet connection"
        echo "  - Verify Dockerfile is correct"
        exit 1
    fi

    log_success "Docker image built"
else
    log_success "Docker image ready"
fi

echo ""
log_info "Starting container..."

mkdir -p data logs

docker run -d \
    --name reco-trading \
    --hostname reco-trading-bot \
    --restart unless-stopped \
    -p 127.0.0.1:9000:9000 \
    -v "${ROOT_DIR}/.env:/app/.env:ro" \
    -v "${ROOT_DIR}/data:/app/data" \
    -v "${ROOT_DIR}/logs:/app/logs" \
    -e DASHBOARD_TYPE=web \
    -e RESOURCE_PROFILE="${RESOURCE_PROFILE}" \
    -e WORKERS="${WORKERS}" \
    -e ML_WORKERS="${ML_WORKERS}" \
    -e BINANCE_API_KEY="${BINANCE_API_KEY}" \
    -e BINANCE_API_SECRET="${BINANCE_API_SECRET}" \
    -e BINANCE_TESTNET="${BINANCE_TESTNET}" \
    -e ENVIRONMENT="${ENVIRONMENT}" \
    -e RUNTIME_PROFILE="${RUNTIME_PROFILE}" \
    -e DATABASE_URL="${DATABASE_URL}" \
    -e POSTGRES_DSN="${POSTGRES_DSN:-}" \
    -e MYSQL_DSN="${MYSQL_DSN:-}" \
    -e REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}" \
    -e DASHBOARD_AUTH_ENABLED="${DASHBOARD_AUTH_ENABLED:-true}" \
    -e DASHBOARD_AUTH_MODE="${DASHBOARD_AUTH_MODE:-token}" \
    -e DASHBOARD_USERNAME="${DASHBOARD_USERNAME:-admin}" \
    -e DASHBOARD_PASSWORD="${DASHBOARD_PASSWORD:-admin}" \
    -e DASHBOARD_API_TOKEN="${DASHBOARD_API_TOKEN:-}" \
    -e ENABLE_AUTO_IMPROVER=true \
    -e ENABLE_ML_ENGINE=true \
    -e ENABLE_CONTINUAL_LEARNING=true \
    -e ENABLE_META_LEARNING=true \
    -e ENABLE_DRIFT_DETECTION=true \
    -e ENABLE_ONCHAIN_ANALYSIS=true \
    -e ENABLE_EVOLUTION=true \
    -e ENABLE_TFT=true \
    -e ENABLE_NBEATS=true \
    -e ENABLE_ADVANCED_META_LEARNING=true \
    -e ENABLE_REINFORCEMENT_LEARNING=true \
    --memory="${RAM_GB}g" \
    --cpus="${CPU_CORES}" \
    reco-trading:latest 2>&1 || {
        log_error "Failed to start container"
        echo ""
        echo "Check logs with: docker logs reco-trading"
        exit 1
    }

log_success "Container started"

echo ""
log_info "Waiting for service..."

DASHBOARD_READY=false
MAX_RETRIES=60
RETRY_COUNT=0

while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
    if ! docker ps --format '{{.Names}}' | grep -q "^reco-trading$"; then
        log_error "Container stopped unexpectedly"
        echo ""
        echo "=== Container Logs ==="
        docker logs reco-trading 2>&1 | tail -50
        exit 1
    fi
    
    if curl -sf http://localhost:9000/api/health >/dev/null 2>&1; then
        DASHBOARD_READY=true
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 2
done

echo ""

if [[ "$DASHBOARD_READY" != "true" ]]; then
    log_error "Dashboard not responding after $((MAX_RETRIES * 2)) seconds"
    echo ""
    echo "=== Container Logs ==="
    docker logs reco-trading 2>&1 | tail -50
    exit 1
fi

log_success "Dashboard available at http://localhost:9000"

if check_command cloudflared; then
    log_info "Starting Cloudflare Tunnel..."
    
    pkill -f "cloudflared tunnel" 2>/dev/null || true
    sleep 1
    
    cloudflared tunnel --url http://localhost:9000 2>&1 | tee /tmp/cloudflared_output.log &
    CLOUDFLARED_PID=$!
    
    MAX_CF_RETRIES=30
    CF_RETRY_COUNT=0
    
    while [[ $CF_RETRY_COUNT -lt $MAX_CF_RETRIES ]]; do
        CLOUDFLARED_URL=$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' /tmp/cloudflared_output.log 2>/dev/null | head -1)
        
        if [[ -z "$CLOUDFLARED_URL" ]]; then
            CLOUDFLARED_URL=$(grep -oE 'https://[a-zA-Z0-9.-]+\.cloudflare\.com' /tmp/cloudflared_output.log 2>/dev/null | head -1)
        fi
        
        if [[ -n "$CLOUDFLARED_URL" ]]; then
            break
        fi
        
        if ! kill -0 "$CLOUDFLARED_PID" 2>/dev/null; then
            log_warn "cloudflared exited unexpectedly"
            break
        fi
        
        CF_RETRY_COUNT=$((CF_RETRY_COUNT + 1))
        echo -n "."
        sleep 2
    done
    
    echo ""
    
    if [[ -n "$CLOUDFLARED_URL" ]]; then
        log_success "Tunnel cloudflared: $CLOUDFLARED_URL"
    else
        log_warn "cloudflared started but no public URL"
    fi
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  SERVICE STARTED SUCCESSFULLY${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

echo -e "${CYAN}Dashboard:${NC} http://localhost:9000"

if [[ -n "$CLOUDFLARED_URL" ]]; then
    echo -e "${CYAN}Public:${NC} $CLOUDFLARED_URL"
fi

echo ""
echo -e "${YELLOW}Status:${NC}"
echo "  Container: $(docker ps --filter name=reco-trading --format '{{.Status}}')"
echo "  Profile: ${RESOURCE_PROFILE}"
echo "  Testnet: ${BINANCE_TESTNET}"

echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "  docker logs reco-trading        - View logs"
echo "  docker logs -f reco-trading     - Follow logs"
echo "  docker stop reco-trading        - Stop container"
echo "  docker restart reco-trading     - Restart container"
echo "  docker exec -it reco-trading bash - Shell into container"

echo ""
CLEANUP_ON_EXIT=false
echo "Press Ctrl+C to stop the service and exit"

docker logs -f reco-trading 2>&1
