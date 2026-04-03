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

# ============================================
# TRAP FOR CLEANUP
# ============================================

CLEANUP_ON_EXIT=true
CLOUDFLARED_PID=""
DOCKER_RUNNING=true

cleanup() {
    if [[ "$CLEANUP_ON_EXIT" == "true" ]]; then
        echo ""
        log_info "Limpiando..."
        
        if [[ -n "$CLOUDFLARED_PID" ]] && kill -0 "$CLOUDFLARED_PID" 2>/dev/null; then
            kill "$CLOUDFLARED_PID" 2>/dev/null || true
            log_info "Cloudflare Tunnel detenido"
        fi
        
        log_info "Contenedor sigue ejecutándose en background"
    fi
}

trap cleanup SIGINT SIGTERM EXIT

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Reco-Trading Docker Builder v4.0${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# ============================================
# CHECK PREREQUISITES
# ============================================

log_info "Verificando prerrequisitos..."

check_command() {
    command -v "$1" >/dev/null 2>&1
}

# ============================================
# DETECT SYSTEM RESOURCES
# ============================================

log_info "Detectando recursos del sistema..."

CPU_CORES=2
if [[ -f /proc/cpuinfo ]]; then
    CPU_CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "2")
elif [[ -f /usr/sbin/sysctl ]]; then
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
elif [[ -f /usr/sbin/sysctl ]]; then
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
    handle_error "Memoria insuficiente (${RAM_GB}GB). Mínimo 1GB requerido."
    exit 1
fi

# Determine profile
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

log_info "Perfil: ${RESOURCE_PROFILE}"

# ============================================
# CHECK DOCKER
# ============================================

log_info "Verificando Docker..."

if ! check_command docker; then
    log_error "Docker no encontrado"
    echo ""
    echo "Instala Docker desde: https://docs.docker.com/get-docker/"
    exit 1
fi

log_success "Docker instalado"

# Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    log_error "Docker daemon no está ejecutándose"
    echo ""
    echo "Inicia Docker con:"
    echo "  sudo systemctl start docker"
    echo ""
    echo "O en Windows/Mac: inicia Docker Desktop"
    exit 1
fi

log_success "Docker daemon activo"

# ============================================
# CHECK DOCKER COMPOSE
# ============================================

if ! docker compose version >/dev/null 2>&1 && ! docker-compose --version >/dev/null 2>&1; then
    log_warn "docker compose no disponible"
    echo ""
    echo "El instalador usará comandos individuales de docker"
fi

# ============================================
# CHECK .ENV FILE
# ============================================

log_info "Verificando configuración..."

if [[ ! -f ".env" ]]; then
    log_error ".env no encontrado"
    echo ""
    echo "Crea un archivo .env con tus credenciales:"
    echo "  BINANCE_API_KEY=tu_api_key"
    echo "  BINANCE_API_SECRET=tu_api_secret"
    echo "  BINANCE_TESTNET=true"
    exit 1
fi

# Load .env
set -a
source .env 2>/dev/null || true
set +a

# Validate API keys
if [[ -z "${BINANCE_API_KEY:-}" ]] || [[ "$BINANCE_API_KEY" == "CAMBIAR_POR_TU_API_KEY" ]]; then
    log_error "BINANCE_API_KEY no configurada en .env"
    exit 1
fi

if [[ -z "${BINANCE_API_SECRET:-}" ]] || [[ "$BINANCE_API_SECRET" == "CAMBIAR_POR_TU_API_SECRET" ]]; then
    log_error "BINANCE_API_SECRET no configurada en .env"
    exit 1
fi

log_success ".env verificado"

# ============================================
# HANDLE EXISTING CONTAINER AND IMAGE
# ============================================

IMAGE_EXISTS=false
if docker image ls reco-trading:latest -q 2>/dev/null | grep -q .; then
    IMAGE_EXISTS=true
fi

CONTAINER_EXISTS=false
if docker ps -a --format '{{.Names}}' | grep -q "^reco-trading$"; then
    CONTAINER_EXISTS=true
fi

if [[ "$IMAGE_EXISTS" == "true" ]]; then
    log_warn "Imagen Docker 'reco-trading:latest' ya existe"
    echo ""
    echo "Opciones:"
    echo "  1) Usar imagen existente y crear contenedor"
    echo "  2) Eliminar imagen y reconstruir desde cero"
    echo "  3) Actualizar solo el contenedor (recrear)"
    echo ""
    read -p "Elige opción (1/2/3): " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            log_info "Usando imagen existente"
            USE_EXISTING_IMAGE=true
            ;;
        2)
            log_info "Eliminando imagen anterior..."
            docker rmi reco-trading:latest 2>/dev/null || true
            USE_EXISTING_IMAGE=false
            ;;
        3)
            log_info "Actualizando contenedor..."
            if [[ "$CONTAINER_EXISTS" == "true" ]]; then
                docker stop reco-trading 2>/dev/null || true
                docker rm reco-trading 2>/dev/null || true
            fi
            USE_EXISTING_IMAGE=true
            ;;
        *)
            log_info "Usando imagen existente por defecto"
            USE_EXISTING_IMAGE=true
            ;;
    esac
elif [[ "$CONTAINER_EXISTS" == "true" ]]; then
    log_warn "Contenedor anterior encontrado"
    echo ""
    echo "Opciones:"
    echo "  1) Detener y eliminar contenedor anterior"
    echo "  2) Mantener contenedor y salir"
    echo "  3) Solo actualizar imagen"
    echo ""
    read -p "Elige opción (1/2/3): " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            log_info "Deteniendo contenedor anterior..."
            docker stop reco-trading 2>/dev/null || true
            docker rm reco-trading 2>/dev/null || true
            log_success "Contenedor anterior eliminado"
            USE_EXISTING_IMAGE=false
            ;;
        2)
            log_info "Manteniendo contenedor actual"
            CLEANUP_ON_EXIT=false
            docker start reco-trading 2>/dev/null || true
            log_success "Contenedor iniciado"
            exit 0
            ;;
        3)
            log_info "Solo actualizando imagen..."
            docker stop reco-trading 2>/dev/null || true
            docker rm reco-trading 2>/dev/null || true
            USE_EXISTING_IMAGE=false
            ;;
        *)
            log_info "Usando opción por defecto"
            USE_EXISTING_IMAGE=true
            ;;
    esac
else
    USE_EXISTING_IMAGE=false
fi

# ============================================
# BUILD OR USE EXISTING DOCKER IMAGE
# ============================================

echo ""
log_info "Preparando imagen Docker..."

# Default values
BINANCE_TESTNET="${BINANCE_TESTNET:-true}"
ENVIRONMENT="${ENVIRONMENT:-testnet}"
RUNTIME_PROFILE="${RUNTIME_PROFILE:-paper}"
DATABASE_URL="${DATABASE_URL:-sqlite:///./data/reco_trading.db}"

echo "  Profile: ${RESOURCE_PROFILE}"
echo "  Environment: ${ENVIRONMENT}"
echo "  Testnet: ${BINANCE_TESTNET}"

if [[ "${USE_EXISTING_IMAGE:-false}" == "true" ]] && [[ "$IMAGE_EXISTS" == "true" ]]; then
    log_info "Usando imagen Docker existente..."
    
    # Verify image still exists
    if docker image ls reco-trading:latest -q 2>/dev/null | grep -q .; then
        log_success "Imagen existente verificada"
    else
        log_warn "La imagen ya no existe, construyendo nueva..."
        USE_EXISTING_IMAGE=false
    fi
fi

if [[ "${USE_EXISTING_IMAGE:-false}" != "true" ]]; then
    # Build with error handling
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
        --build-arg ENABLE_AUTO_IMPROVER=true \
        --build-arg ENABLE_ML_ENGINE=true \
        --build-arg ENABLE_CONTINUAL_LEARNING=true \
        --build-arg ENABLE_META_LEARNING=true \
        --build-arg ENABLE_DRIFT_DETECTION=true \
        --build-arg ENABLE_ONCHAIN_ANALYSIS=true \
        --build-arg ENABLE_EVOLUTION=true \
        --build-arg ENABLE_TFT=true \
        --build-arg ENABLE_NBEATS=true \
        --build-arg ENABLE_ADVANCED_META_LEARNING=true \
        --build-arg ENABLE_REINFORCEMENT_LEARNING=true \
        -t reco-trading:latest . 2>&1 || BUILD_FAILED=true

    if [[ "$BUILD_FAILED" == "true" ]]; then
        log_error "Error al construir imagen Docker"
        echo ""
        echo "Revisa los errores arriba e intenta:"
        echo "  - Verificar que Docker tiene recursos suficientes"
        echo "  - Verificar conexión a internet"
        echo "  - Verificar que el Dockerfile es correcto"
        exit 1
    fi

    log_success "Imagen Docker construida"
else
    log_success "Imagen Docker preparada"
fi

# ============================================
# RUN CONTAINER
# ============================================

echo ""
log_info "Iniciando contenedor..."

# Create directories
mkdir -p data logs

# Run container
docker run -d \
    --name reco-trading \
    --hostname reco-trading-bot \
    -p 127.0.0.1:9000:9000 \
    -v "${ROOT_DIR}/.env:/app/.env:ro" \
    -v "${ROOT_DIR}/data:/app/data" \
    -v "${ROOT_DIR}/logs:/app/logs" \
    -e DASHBOARD_TYPE=web \
    -e RESOURCE_PROFILE="${RESOURCE_PROFILE}" \
    -e WORKERS="${WORKERS}" \
    -e ML_WORKERS="${ML_WORKERS}" \
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
    --restart unless-stopped \
    --memory="${RAM_GB}g" \
    --cpus="${CPU_CORES}" \
    reco-trading:latest 2>&1 || {
        log_error "Error al iniciar contenedor"
        echo ""
        echo "Revisa los logs con: docker logs reco-trading"
        exit 1
    }

log_success "Contenedor iniciado"

# ============================================
# WAIT FOR SERVICE
# ============================================

echo ""
log_info "Esperando que el servicio esté disponible..."

DASHBOARD_READY=false
MAX_RETRIES=60
RETRY_COUNT=0

while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^reco-trading$"; then
        log_error "Contenedor dejó de ejecutarse"
        echo ""
        echo "=== Logs del contenedor ==="
        docker logs reco-trading 2>&1 | tail -50
        exit 1
    fi
    
    # Try health endpoint
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
    log_error "Dashboard no respondió después de $((MAX_RETRIES * 2)) segundos"
    echo ""
    echo "=== Logs del contenedor ==="
    docker logs reco-trading 2>&1 | tail -50
    exit 1
fi

log_success "Dashboard disponible en http://localhost:9000"

# ============================================
# CLOUDFLARE TUNNEL (OPTIONAL)
# ============================================

CLOUDFLARED_URL=""

if check_command cloudflared; then
    log_info "Iniciando Cloudflare Tunnel..."
    
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
            log_warn "cloudflared terminó inesperadamente"
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
        log_warn "cloudflared iniciado pero sin URL pública"
    fi
fi

# ============================================
# FINAL OUTPUT
# ============================================

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ SERVICIO INICIADO${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

echo -e "${CYAN}📊 Dashboard:${NC} http://localhost:9000"

if [[ -n "$CLOUDFLARED_URL" ]]; then
    echo -e "${CYAN}🌐 Público:${NC} $CLOUDFLARED_URL"
fi

echo ""
echo -e "${YELLOW}ℹ️  Estado:${NC}"
echo "  - Contenedor: $(docker ps --filter name=reco-trading --format '{{.Status}}')"
echo "  - Profile: ${RESOURCE_PROFILE}"
echo "  - Testnet: ${BINANCE_TESTNET}"

echo ""
echo -e "${YELLOW}ℹ️  Comandos útiles:${NC}"
echo "  docker logs reco-trading - Ver logs"
echo "  docker stop reco-trading - Detener"
echo "  docker restart reco-trading - Reiniciar"

echo ""
CLEANUP_ON_EXIT=false
echo "Presiona Ctrl+C para detener el servicios y salir"

# Follow logs
docker logs -f reco-trading 2>&1
