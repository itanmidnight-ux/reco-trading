#!/bin/bash
#
# Reco-Trading Docker Build & Run Script
# =========================================
# Builds and runs Reco-Trading bot in Docker with:
# - Auto system detection and resource optimization
# - Web Dashboard (port 9000)
# - Cloudflare Tunnel for public access
# - Auto-reads .env from local
# - Works without root/sudo if possible
# - All ML and auto-improvement features enabled
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
echo -e "${CYAN}  Reco-Trading Docker Builder v4.0${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# ============================================
# DETECT SYSTEM RESOURCES
# ============================================

log_info "Detectando recursos del sistema..."

# Detect CPU cores
if [[ -f /proc/cpuinfo ]]; then
  CPU_CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "2")
elif [[ -f /usr/sbin/sysctl ]]; then
  CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "2")
else
  CPU_CORES="2"
fi

# Detect RAM (in GB)
if [[ -f /proc/meminfo ]]; then
  TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
  RAM_GB=$((TOTAL_MEM / 1024 / 1024))
elif [[ -f /usr/sbin/sysctl ]]; then
  RAM_GB=$(sysctl -n hw.memsize 2>/dev/null || echo "2048")
  RAM_GB=$((RAM_GB / 1024 / 1024 / 1024))
else
  RAM_GB="2"
fi

# Detect disk space (in GB)
if [[ -f /proc/mounts ]]; then
  DISK_GB=$(df -BG / 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
else
  DISK_GB="10"
fi

echo -e "  ${GREEN}CPU Cores: ${CPU_CORES}${NC}"
echo -e "  ${GREEN}RAM: ${RAM_GB} GB${NC}"
echo -e "  ${GREEN}Disk: ${DISK_GB} GB${NC}"

# Determine resource profile
if [[ $RAM_GB -ge 16 ]] && [[ $CPU_CORES -ge 4 ]]; then
  RESOURCE_PROFILE="high"
  WORKERS="4"
  ML_WORKERS="4"
  log_info "Perfil: ALTO RENDIMIENTO"
elif [[ $RAM_GB -ge 8 ]] && [[ $CPU_CORES -ge 2 ]]; then
  RESOURCE_PROFILE="medium"
  WORKERS="2"
  ML_WORKERS="2"
  log_info "Perfil: RENDIMIENTO MEDIO"
else
  RESOURCE_PROFILE="low"
  WORKERS="1"
  ML_WORKERS="1"
  log_info "Perfil: BAJO RENDIMIENTO (optimizado)"
fi

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
echo -e "${YELLOW}Perfil de recursos: ${RESOURCE_PROFILE}${NC}"

# ============================================
# BUILD DOCKER IMAGE
# ============================================

echo ""
log_info "Construyendo imagen Docker con optimización..."

# Build with all environment variables and resource-based settings
docker build \
  --build-arg RESOURCE_PROFILE="${RESOURCE_PROFILE}" \
  --build-arg WORKERS="${WORKERS}" \
  --build-arg ML_WORKERS="${ML_WORKERS}" \
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
  -t reco-trading:latest . 2>&1

if [[ $? -eq 0 ]]; then
  log_success "Imagen Docker construida"
else
  log_error "Error al construir imagen Docker"
  exit 1
fi

# ============================================
# RUN CONTAINER
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

# Run container in detached mode with all features enabled
docker run -d \
  --name reco-trading \
  --hostname reco-trading-bot \
  -p 9000:9000 \
  -p 8080:8080 \
  -v "${SCRIPT_DIR}/.env:/app/.env:ro" \
  -v "${SCRIPT_DIR}/data:/app/data" \
  -v "${SCRIPT_DIR}/logs:/app/logs" \
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
  -e CONFIRM_MAINNET="${CONFIRM_MAINNET:-false}" \
  -e ENVIRONMENT="${ENVIRONMENT}" \
  -e RUNTIME_PROFILE="${RUNTIME_PROFILE}" \
  -e POSTGRES_DSN="${POSTGRES_DSN:-}" \
  -e MYSQL_DSN="${MYSQL_DSN:-}" \
  -e DATABASE_URL="${DATABASE_URL:-}" \
  -e REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}" \
  --restart unless-stopped \
  --memory="${RAM_GB}g" \
  --cpus="${CPU_CORES}" \
  reco-trading:latest 2>&1

if [[ $? -eq 0 ]]; then
  log_success "Contenedor iniciado"
else
  log_error "Error al iniciar contenedor"
  exit 1
fi

# ============================================
# WAIT FOR WEB DASHBOARD
# ============================================

echo ""
log_info "Esperando que el dashboard web esté disponible..."

DASHBOARD_READY=false
MAX_RETRIES=60
RETRY_COUNT=0

while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
  if curl -sf http://localhost:9000/api/health >/dev/null 2>&1; then
    DASHBOARD_READY=true
    break
  fi
  
  # Check if container is still running
  if ! docker ps --format '{{.Names}}' | grep -q "^reco-trading$"; then
    log_error "Contenedor dejó de ejecutarse"
    echo ""
    echo "=== Logs del contenedor ==="
    docker logs reco-trading 2>&1 | tail -80
    exit 1
  fi
  
  RETRY_COUNT=$((RETRY_COUNT + 1))
  echo -n "."
  sleep 2
done

echo ""

if [[ "$DASHBOARD_READY" != "true" ]]; then
  log_error "Dashboard web no respondió después de $((MAX_RETRIES * 2)) segundos"
  echo ""
  echo "=== Logs del contenedor ==="
  docker logs reco-trading 2>&1 | tail -80
  exit 1
fi

log_success "Dashboard web disponible en http://localhost:9000"

# ============================================
# INSTALL CLOUDFLARED IF NEEDED
# ============================================

log_info "Verificando cloudflared..."

CLOUDFLARED_CMD=""

if command -v cloudflared >/dev/null 2>&1; then
  CLOUDFLARED_CMD="cloudflared"
  log_success "cloudflared encontrado"
elif [[ "$IS_ROOT" == "true" ]] || [[ "$IS_SUDO" == "true" ]]; then
  log_info "Instalando cloudflared..."
  
  if [[ "$(uname -m)" == "x86_64" ]]; then
    CF_ARCH="amd64"
  elif [[ "$(uname -m)" == "aarch64" ]]; then
    CF_ARCH="arm64"
  else
    CF_ARCH="amd64"
  fi
  
  if check_command curl; then
    ${RUN_CMD} curl -sSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CF_ARCH}" -o /usr/local/bin/cloudflared 2>/dev/null
  elif check_command wget; then
    ${RUN_CMD} wget -q "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CF_ARCH}" -O /usr/local/bin/cloudflared 2>/dev/null
  fi
  
  if [[ -f /usr/local/bin/cloudflared ]]; then
    ${RUN_CMD} chmod +x /usr/local/bin/cloudflared
    CLOUDFLARED_CMD="cloudflared"
    log_success "cloudflared instalado"
  else
    log_warn "No se pudo instalar cloudflared automáticamente"
  fi
else
  log_warn "cloudflared no encontrado y sin permisos para instalar"
  log_info "Instala cloudflared manualmente: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
fi

# ============================================
# START CLOUDFLARED TUNNEL
# ============================================

CLOUDFLARED_URL=""

if [[ -n "$CLOUDFLARED_CMD" ]]; then
  log_info "Iniciando Cloudflare Tunnel..."
  echo ""
  
  # Kill any previous cloudflared processes for this tunnel
  pkill -f "cloudflared tunnel --url http://localhost:9000" 2>/dev/null || true
  sleep 1
  
  # Start cloudflared tunnel in background, capture output
  $CLOUDFLARED_CMD tunnel --url http://localhost:9000 2>&1 | tee /tmp/cloudflared_output.log &
  CLOUDFLARED_PID=$!
  
  # Wait for cloudflared to establish tunnel and get URL
  log_info "Esperando tunnel cloudflared..."
  
  MAX_CF_RETRIES=30
  CF_RETRY_COUNT=0
  
  while [[ $CF_RETRY_COUNT -lt $MAX_CF_RETRIES ]]; do
    # Try multiple patterns to extract the URL
    CLOUDFLARED_URL=$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' /tmp/cloudflared_output.log 2>/dev/null | head -1)
    
    if [[ -z "$CLOUDFLARED_URL" ]]; then
      CLOUDFLARED_URL=$(grep -oE 'https://[a-zA-Z0-9.-]+\.cloudflare\.com' /tmp/cloudflared_output.log 2>/dev/null | head -1)
    fi
    
    if [[ -z "$CLOUDFLARED_URL" ]]; then
      CLOUDFLARED_URL=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' /tmp/cloudflared_output.log 2>/dev/null | tail -1)
    fi
    
    if [[ -n "$CLOUDFLARED_URL" ]]; then
      break
    fi
    
    # Check if cloudflared is still running
    if ! kill -0 $CLOUDFLARED_PID 2>/dev/null; then
      log_error "cloudflared terminó inesperadamente"
      cat /tmp/cloudflared_output.log 2>/dev/null
      exit 1
    fi
    
    CF_RETRY_COUNT=$((CF_RETRY_COUNT + 1))
    echo -n "."
    sleep 2
  done
  
  echo ""
  
  if [[ -n "$CLOUDFLARED_URL" ]]; then
    log_success "Tunnel cloudflared establecido"
  else
    log_warn "No se pudo obtener URL de cloudflared (el tunnel puede estar activo sin URL capturada)"
    echo ""
    echo "=== Salida de cloudflared ==="
    cat /tmp/cloudflared_output.log 2>/dev/null | tail -20
  fi
else
  log_warn "cloudflared no disponible - sin tunnel público"
fi

# ============================================
# FINAL OUTPUT
# ============================================

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ SERVICIO INICIADO CON ÉXITO${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${CYAN}📊 Dashboard Web Local:${NC}"
echo -e "  ${GREEN}http://localhost:9000${NC}"
echo ""

if [[ -n "$CLOUDFLARED_URL" ]]; then
  echo -e "${CYAN}🌐 URL de Acceso Público (Cloudflare Tunnel):${NC}"
  echo ""
  echo -e "  ${GREEN}${CLOUDFLARED_URL}${NC}"
  echo ""
fi

echo -e "${YELLOW}ℹ️  Características activas:${NC}"
echo "  ✓ Machine Learning (ML Engine)"
echo "  ✓ Auto-Improver (Mejora automática)"
echo "  ✓ Continual Learning (Aprendizaje continuo)"
echo "  ✓ Meta-Learning (Meta-aprendizaje)"
echo "  ✓ Drift Detection (Detección de cambios)"
echo "  ✓ On-Chain Analysis (Análisis en cadena)"
echo "  ✓ Evolution (Evolución de estrategias)"
echo "  ✓ Temporal Fusion Transformer (TFT)"
echo "  ✓ N-BEATS"
echo "  ✓ Advanced Meta-Learning"
echo "  ✓ Reinforcement Learning (PPO/TD3)"
echo ""
echo -e "${YELLOW}ℹ️  Notas:${NC}"
echo "  - Dashboard web disponible en http://localhost:9000"
if [[ -n "$CLOUDFLARED_URL" ]]; then
  echo "  - Acceso público: ${CLOUDFLARED_URL}"
fi
echo "  - El contenedor se reiniciará automáticamente si falla"
echo "  - Presiona Ctrl+C para detener el servicio"
echo ""

# Cleanup on exit
cleanup() {
  echo ""
  log_info "Deteniendo servicios..."
  
  if [[ -n "$CLOUDFLARED_PID" ]] && kill -0 $CLOUDFLARED_PID 2>/dev/null; then
    kill $CLOUDFLARED_PID 2>/dev/null || true
    log_info "Cloudflare Tunnel detenido"
  fi
  
  docker stop reco-trading 2>/dev/null || true
  log_info "Contenedor detenido"
  
  rm -f /tmp/cloudflared_output.log
  exit 0
}

trap cleanup SIGINT SIGTERM

# Follow container logs
echo -e "${BLUE}=== Logs en tiempo real ===${NC}"
echo ""

docker logs -f reco-trading 2>&1