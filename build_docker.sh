#!/bin/bash
#
# Reco-Trading Docker Build & Run Script
# =========================================
# Builds and runs Reco-Trading bot in Docker with:
# - Web Dashboard (port 9000)
# - Cloudflare Tunnel for public access
# - Auto-reads .env from local
# - Works without root/sudo if possible
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Colors
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

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Reco-Trading Docker Builder${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# ============================================
# DETECT ROOT ACCESS
# ============================================

IS_ROOT=false
IS_SUDO=false

if [[ "${EUID}" -eq 0 ]]; then
  IS_ROOT=true
  log_info "Ejecutando como ROOT"
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  IS_SUDO=true
  log_info "SUDO disponible sin contraseña"
else
  log_warn "Sin acceso root/sudo - modo usuario"
fi

RUN_CMD=""
if [[ "$IS_ROOT" == "true" ]]; then
  RUN_CMD=""
elif [[ "$IS_SUDO" == "true" ]]; then
  RUN_CMD="sudo"
fi

# ============================================
# CHECK DOCKER
# ============================================

check_command() {
  command -v "$1" >/dev/null 2>&1
}

log_info "Verificando Docker..."

if check_command docker; then
  log_success "Docker instalado"
  
  # Check if Docker daemon is running
  if docker info >/dev/null 2>&1; then
    log_success "Docker daemon activo"
  else
    log_error "Docker instalado pero el daemon no está ejecutándose"
    if [[ "$IS_ROOT" == "true" ]] || [[ "$IS_SUDO" == "true" ]]; then
      log_info "Iniciando Docker daemon..."
      ${RUN_CMD} systemctl start docker 2>/dev/null || ${RUN_CMD} dockerd &>/dev/null &
      sleep 3
      if docker info >/dev/null 2>&1; then
        log_success "Docker daemon iniciado"
      else
        log_error "No se pudo iniciar Docker daemon"
        exit 1
      fi
    else
      log_error "Inicia Docker daemon manualmente o usa sudo"
      exit 1
    fi
  fi
else
  log_warn "Docker no encontrado"
  
  if [[ "$IS_ROOT" == "true" ]] || [[ "$IS_SUDO" == "true" ]]; then
    log_info "Instalando Docker..."
    
    # Detect OS
    if check_command apt-get; then
      ${RUN_CMD} apt-get update -qq
      ${RUN_CMD} apt-get install -y -qq apt-transport-https ca-certificates curl gnupg lsb-release
      curl -fsSL https://download.docker.com/linux/debian/gpg | ${RUN_CMD} gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
      echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | ${RUN_CMD} tee /etc/apt/sources.list.d/docker.list >/dev/null
      ${RUN_CMD} apt-get update -qq
      ${RUN_CMD} apt-get install -y -qq docker-ce docker-ce-cli containerd.io
      log_success "Docker instalado"
      
      # Start Docker
      ${RUN_CMD} systemctl start docker || ${RUN_CMD} dockerd &>/dev/null &
      sleep 3
    elif check_command yum; then
      ${RUN_CMD} yum install -y yum-utils
      ${RUN_CMD} yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
      ${RUN_CMD} yum install -y docker-ce docker-ce-cli containerd.io
      ${RUN_CMD} systemctl start docker
    else
      log_error "No se puede instalar Docker automáticamente"
      log_info "Por favor instala Docker manualmente desde https://docs.docker.com/get-docker/"
      exit 1
    fi
  else
    log_error "Docker no instalado y sin permisos para instalar"
    log_info "Instala Docker manualmente: https://docs.docker.com/get-docker/"
    exit 1
  fi
fi

# ============================================
# LOAD .ENV FROM LOCAL
# ============================================

log_info "Cargando configuración local..."

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
  log_success ".env cargado"
  
  # Show loaded config (masked)
  if [[ -n "${BINANCE_API_KEY:-}" ]] && [[ "${BINANCE_API_KEY}" != "CAMBIAR_POR_TU_API_KEY" ]]; then
    echo -e "  ${GREEN}✓ BINANCE_API_KEY configurada${NC}"
  else
    echo -e "  ${RED}✗ BINANCE_API_KEY no configurada${NC}"
  fi
  
  if [[ -n "${BINANCE_API_SECRET:-}" ]] && [[ "${BINANCE_API_SECRET}" != "CAMBIAR_POR_TU_API_SECRET" ]]; then
    echo -e "  ${GREEN}✓ BINANCE_API_SECRET configurada${NC}"
  else
    echo -e "  ${RED}✗ BINANCE_API_SECRET no configurada${NC}"
  fi
else
  log_error ".env no encontrado"
  echo ""
  echo "Crea un archivo .env con tus credenciales:"
  echo "  BINANCE_API_KEY=tu_api_key"
  echo "  BINANCE_API_SECRET=tu_api_secret"
  echo "  BINANCE_TESTNET=true"
  exit 1
fi

# Validate API keys
if [[ -z "${BINANCE_API_KEY:-}" ]] || [[ "$BINANCE_API_KEY" == "CAMBIAR_POR_TU_API_KEY" ]]; then
  log_error "BINANCE_API_KEY no configurada en .env"
  exit 1
fi

if [[ -z "${BINANCE_API_SECRET:-}" ]] || [[ "$BINANCE_API_SECRET" == "CAMBIAR_POR_TU_API_SECRET" ]]; then
  log_error "BINANCE_API_SECRET no configurada en .env"
  exit 1
fi

# ============================================
# PREPARE DOCKER ENVIRONMENT VARIABLES
# ============================================

# Use testnet by default unless specified
BINANCE_TESTNET="${BINANCE_TESTNET:-true}"
ENVIRONMENT="${ENVIRONMENT:-testnet}"
RUNTIME_PROFILE="${RUNTIME_PROFILE:-paper}"

# Database fallback
if [[ -z "${POSTGRES_DSN:-}" ]] && [[ -z "${MYSQL_DSN:-}" ]]; then
  DATABASE_URL="${DATABASE_URL:-sqlite:///./data/reco_trading.db}"
  log_warn "Sin PostgreSQL/MySQL - usando SQLite"
fi

echo ""
echo -e "${YELLOW}Modo: ${ENVIRONMENT} (testnet: ${BINANCE_TESTNET})${NC}"

# ============================================
# BUILD DOCKER IMAGE
# ============================================

echo ""
log_info "Construyendo imagen Docker..."

# Build with all environment variables
docker build \
  --build-arg BINANCE_API_KEY="${BINANCE_API_KEY}" \
  --build-arg BINANCE_API_SECRET="${BINANCE_API_SECRET}" \
  --build-arg BINANCE_TESTNET="${BINANCE_TESTNET}" \
  --build-arg CONFIRM_MAINNET="${CONFIRM_MAINNET:-false}" \
  --build-arg ENVIRONMENT="${ENVIRONMENT}" \
  --build-arg RUNTIME_PROFILE="${RUNTIME_PROFILE}" \
  --build-arg POSTGRES_DSN="${POSTGRES_DSN:-}" \
  --build-arg POSTGRES_ADMIN_DSN="${POSTGRES_ADMIN_DSN:-}" \
  --build-arg MYSQL_DSN="${MYSQL_DSN:-}" \
  --build-arg DATABASE_URL="${DATABASE_URL:-}" \
  --build-arg REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}" \
  -t reco-trading:latest . 2>&1

if [[ $? -eq 0 ]]; then
  log_success "Imagen Docker construida"
else
  log_error "Error al construir imagen Docker"
  exit 1
fi

# ============================================
# RUN CONTAINER WITH CLOUDFLARED TUNNEL
# ============================================

echo ""
log_info "Iniciando contenedor..."

# Stop existing container if any
if docker ps -a --format '{{.Names}}' | grep -q "^reco-trading$"; then
  log_warn "Contenedor anterior encontrado, deteniendo..."
  docker stop reco-trading 2>/dev/null || true
  docker rm reco-trading 2>/dev/null || true
fi

# Create data directory if not exists
mkdir -p data logs

# Run container in detached mode with .env mounted
# And start cloudflared tunnel
docker run -d \
  --name reco-trading \
  -p 9000:9000 \
  -p 8080:8080 \
  -v "${SCRIPT_DIR}/.env:/app/.env:ro" \
  -v "${SCRIPT_DIR}/data:/app/data" \
  -v "${SCRIPT_DIR}/logs:/app/logs" \
  -e DASHBOARD_TYPE=web \
  -e BINANCE_API_KEY="${BINANCE_API_KEY}" \
  -e BINANCE_API_SECRET="${BINANCE_API_SECRET}" \
  -e BINANCE_TESTNET="${BINANCE_TESTNET}" \
  -e CONFIRM_MAINNET="${CONFIRM_MAINNET:-false}" \
  -e ENVIRONMENT="${ENVIRONMENT}" \
  -e RUNTIME_PROFILE="${RUNTIME_PROFILE}" \
  -e POSTGRES_DSN="${POSTGRES_DSN:-}" \
  -e MYSQL_DSN="${MYSQL_DSN:-}" \
  -e DATABASE_URL="${DATABASE_URL:-}" \
  -e REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}" \
  --restart unless-stopped \
  reco-trading:latest 2>&1

if [[ $? -eq 0 ]]; then
  log_success "Contenedor iniciado"
else
  log_error "Error al iniciar contenedor"
  exit 1
fi

# Wait for container to start and get cloudflared URL
echo ""
log_info "Esperando que el servicio esté disponible..."

# Wait for container to be running
sleep 5

# Get logs to find cloudflared URL
log_info "Obteniendo URL del tunnel..."

# Try to get the cloudflared URL from container logs
CLOUDFLARED_URL=""
MAX_RETRIES=30
RETRY_COUNT=0

while [[ -z "$CLOUDFLARED_URL" ]] && [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
  # Check if container is still running
  if ! docker ps --format '{{.Names}}' | grep -q "^reco-trading$"; then
    log_error "Contenedor dejó de ejecutarse"
    echo ""
    echo "=== Logs del contenedor ==="
    docker logs reco-trading 2>&1 | tail -50
    exit 1
  fi
  
  # Try to get tunnel URL from cloudflared
  CLOUDFLARED_URL=$(docker logs reco-trading 2>&1 | grep -oE 'https://[^ ]+\.trycloudflare\.com' | head -1)
  
  if [[ -z "$CLOUDFLARED_URL" ]]; then
    # Also check for other tunnel providers or custom format
    CLOUDFLARED_URL=$(docker logs reco-trading 2>&1 | grep -oE 'https://[a-zA-Z0-9.-]+\.cloudflared\.io' | head -1)
  fi
  
  if [[ -z "$CLOUDFLARED_URL" ]]; then
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 2
  fi
done

echo ""

if [[ -n "$CLOUDFLARED_URL" ]]; then
  echo ""
  echo -e "${GREEN}========================================${NC}"
  echo -e "${GREEN}  ✅ SERVICIO INICIADO${NC}"
  echo -e "${GREEN}========================================${NC}"
  echo ""
  echo -e "${CYAN}🌐 URL DE ACCESO PÚBLICO:${NC}"
  echo ""
  echo -e "  ${GREEN}${CLOUDFLARED_URL}${NC}"
  echo ""
  echo -e "${YELLOW}ℹ️  Notas:${NC}"
  echo "  - El dashboard web está disponible en esa URL"
  echo "  - El tunnel se reiniciará automáticamente si se desconecta"
  echo "  - Presiona Ctrl+C para detener el servicio"
  echo ""
  echo -e "${BLUE}=== Logs en tiempo real ===${NC}"
  echo ""
  
  # Follow container logs
  docker logs -f reco-trading 2>&1
  
else
  log_error "No se pudo obtener la URL del tunnel"
  echo ""
  echo "=== Estado del contenedor ==="
  docker ps -a --filter "name=reco-trading"
  echo ""
  echo "=== Logs del contenedor ==="
  docker logs reco-trading 2>&1 | tail -100
  exit 1
fi