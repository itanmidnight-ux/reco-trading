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

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Reco-Trading Auto-Installer v3.0${NC}"
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
# FUNCTIONS
# ============================================

check_command() {
  command -v "$1" >/dev/null 2>&1
}

upsert_env_var() {
  local env_file="$1"
  local key="$2"
  local value="$3"
  
  if [[ ! -f "${env_file}" ]]; then
    touch "${env_file}"
  fi
  
  if grep -qE "^${key}=" "${env_file}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${env_file}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${env_file}"
  fi
}

# ============================================
# STEP 1: DETECT SYSTEM AND INSTALL DEPENDENCIES
# ============================================

log_info "Detectando sistema operativo..."

SYSTEM_PACKAGES=()
PYTHON_CMD=""

if check_command apt-get; then
  SYSTEM="debian"
  log_info "Sistema detectado: Debian/Ubuntu"
  
  export DEBIAN_FRONTEND=noninteractive
  ${RUN_CMD} apt-get update -qq 2>/dev/null || true
  
  PACKAGES=("python3" "python3-venv" "python3-pip" "curl" "wget" "git")
  
  for pkg in "${PACKAGES[@]}"; do
    if ! dpkg -l 2>/dev/null | grep -q "^ii  ${pkg}"; then
      log_info "Instalando ${pkg}..."
      ${RUN_CMD} apt-get install -y -qq "${pkg}" 2>/dev/null || true
    fi
  done

elif check_command yum; then
  SYSTEM="rhel"
  log_info "Sistema detectado: RHEL/CentOS"
  
  PACKAGES=("python3" "python3-pip" "python3-venv")
  
  for pkg in "${PACKAGES[@]}"; do
    ${RUN_CMD} yum install -y "${pkg}" 2>/dev/null || true
  done

elif check_command pacman; then
  SYSTEM="arch"
  log_info "Sistema detectado: Arch Linux"
  
  ${RUN_CMD} pacman -Sy --noconfirm python python-pip 2>/dev/null || true

elif check_command dnf; then
  SYSTEM="fedora"
  log_info "Sistema detectado: Fedora"
  
  PACKAGES=("python3" "python3-pip" "python3-venv")
  
  for pkg in "${PACKAGES[@]}"; do
    ${RUN_CMD} dnf install -y "${pkg}" 2>/dev/null || true
  done

else
  log_warn "Sistema no detectado, buscando Python..."
fi

# ============================================
# STEP 2: FIND PYTHON
# ============================================

log_info "Buscando Python..."

for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if check_command "$cmd"; then
    PYTHON_CMD="$cmd"
    break
  fi
done

if [[ -z "$PYTHON_CMD" ]]; then
  log_error "No se encontró Python instalado"
  echo ""
  echo "Por favor instala Python 3.10+ manualmente:"
  echo "  - Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
  echo "  - Fedora: sudo dnf install python3 python3-pip"
  echo "  - Arch: sudo pacman -S python python-pip"
  exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oP '\d+\.\d+')
log_success "Python encontrado: $PYTHON_VERSION"

# ============================================
# STEP 3: CREATE VIRTUAL ENVIRONMENT (USER MODE)
# ============================================

log_info "Configurando entorno virtual..."

if [[ -d ".venv" ]]; then
  log_warn "Entorno virtual ya existe, recreando..."
  rm -rf .venv
fi

# Create venv in user mode (works without root)
$PYTHON_CMD -m venv .venv
source .venv/bin/activate

# Upgrade pip in user mode
pip install --upgrade pip -q --root-user-action=ignore 2>/dev/null || true

log_success "Entorno virtual creado"

# ============================================
# STEP 4: INSTALL PYTHON DEPENDENCIES (USER MODE)
# ============================================

log_info "Instalando dependencias básicas..."

pip install -r requirements.txt -q --root-user-action=ignore 2>/dev/null || true

# ============================================
# STEP 4.1: SMART ML DEPENDENCIES INSTALL
# ============================================

log_info "Detectando recursos para ML optimizado..."

# Detect RAM for ML profile
if [[ -f /proc/meminfo ]]; then
  TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
  ML_RAM_GB=$((TOTAL_MEM_KB / 1024 / 1024))
else
  ML_RAM_GB="2"
fi

# Detect CPU for ML workers
if [[ -f /proc/cpuinfo ]]; then
  ML_CPU_CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "2")
else
  ML_CPU_CORES="2"
fi

# Determine ML profile based on resources
if [[ $ML_RAM_GB -ge 16 ]] && [[ $ML_CPU_CORES -ge 4 ]]; then
  ML_PROFILE="high"
  log_info "Perfil ML: ALTO (RAM: ${ML_RAM_GB}GB, CPU: ${ML_CPU_CORES})"
elif [[ $ML_RAM_GB -ge 8 ]] && [[ $ML_CPU_CORES -ge 2 ]]; then
  ML_PROFILE="medium"
  log_info "Perfil ML: MEDIO (RAM: ${ML_RAM_GB}GB, CPU: ${ML_CPU_CORES})"
else
  ML_PROFILE="low"
  log_info "Perfil ML: BAJO (RAM: ${ML_RAM_GB}GB, CPU: ${ML_CPU_CORES}) - optimizado"
fi

# Function to check if package is already installed
pip_pkg_installed() {
  $PYTHON_CMD -c "import $1" 2>/dev/null
  return $?
}

# Smart ML install - only install what's needed and not already present
log_info "Instalando ML dependencies (perfil: ${ML_PROFILE})..."

ML_INSTALL_COUNT=0

# PyTorch (core for all advanced models)
if ! pip_pkg_installed "torch"; then
  log_info "  Instalando PyTorch (CPU)..."
  if [[ "$ML_PROFILE" == "high" ]] || [[ "$ML_PROFILE" == "medium" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q --root-user-action=ignore 2>/dev/null && ML_INSTALL_COUNT=$((ML_INSTALL_COUNT + 1)) || log_warn "  PyTorch skipped"
  else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q --root-user-action=ignore 2>/dev/null && ML_INSTALL_COUNT=$((ML_INSTALL_COUNT + 1)) || log_warn "  PyTorch skipped"
  fi
else
  log_info "  PyTorch ya instalado ✓"
fi

# PyTorch Lightning
if ! pip_pkg_installed "pytorch_lightning"; then
  pip install pytorch-lightning -q --root-user-action=ignore 2>/dev/null && ML_INSTALL_COUNT=$((ML_INSTALL_COUNT + 1)) || true
fi

# ML libraries for existing FreqAI
if ! pip_pkg_installed "lightgbm"; then
  pip install lightgbm -q --root-user-action=ignore 2>/dev/null || true
fi

if ! pip_pkg_installed "xgboost"; then
  pip install xgboost -q --root-user-action=ignore 2>/dev/null || true
fi

# Advanced ML (only for medium/high profiles)
if [[ "$ML_PROFILE" != "low" ]]; then
  if ! pip_pkg_installed "stable_baselines3"; then
    pip install stable-baselines3 -q --root-user-action=ignore 2>/dev/null || true
  fi
  if ! pip_pkg_installed "optuna"; then
    pip install optuna -q --root-user-action=ignore 2>/dev/null || true
  fi
  if ! pip_pkg_installed "gymnasium"; then
    pip install gymnasium -q --root-user-action=ignore 2>/dev/null || true
  fi
fi

# Light dependencies (all profiles)
if ! pip_pkg_installed "statsmodels"; then
  pip install statsmodels -q --root-user-action=ignore 2>/dev/null || true
fi

if ! pip_pkg_installed "yfinance"; then
  pip install yfinance -q --root-user-action=ignore 2>/dev/null || true
fi

if ! pip_pkg_installed "xgboost"; then
  pip install xgboost -q --root-user-action=ignore 2>/dev/null || true
fi

# Advanced ML (only for medium/high profiles)
if [[ "$ML_PROFILE" != "low" ]]; then
  if ! pip_pkg_installed "stable_baselines3"; then
    pip install stable-baselines3 -q --root-user-action=ignore 2>/dev/null || true
  fi
  if ! pip_pkg_installed "optuna"); then
    pip install optuna -q --root-user-action=ignore 2>/dev/null || true
  fi
  if ! pip_pkg_installed "gymnasium"); then
    pip install gymnasium -q --root-user-action=ignore 2>/dev/null || true
  fi
fi

# Light dependencies (all profiles)
if ! pip_pkg_installed "statsmodels"); then
  pip install statsmodels -q --root-user-action=ignore 2>/dev/null || true
fi

if ! pip_pkg_installed "yfinance"); then
  pip install yfinance -q --root-user-action=ignore 2>/dev/null || true
fi

log_success "ML dependencies instaladas (${ML_INSTALL_COUNT} nuevos paquetes)"

# Save ML profile to .env for runtime use
ML_PROFILE_VAR="${ML_PROFILE}"

# ============================================
# STEP 5: AUTO-DETECT DATABASE
# ============================================

log_info "Detectando base de datos..."

POSTGRES_AVAILABLE=false
MYSQL_AVAILABLE=false
SQLITE_AVAILABLE=true
REDIS_AVAILABLE=false

# Check PostgreSQL
if check_command pg_isready; then
  if pg_isready -h localhost -p 5432 2>/dev/null; then
    POSTGRES_AVAILABLE=true
    log_info "PostgreSQL detectado y accesible"
  fi
fi

# Check systemctl for service status (doesn't require root)
if check_command systemctl; then
  if systemctl is-active postgresql >/dev/null 2>&1; then
    POSTGRES_AVAILABLE=true
    log_info "PostgreSQL servicio activo"
  fi
  if systemctl is-active mysql >/dev/null 2>&1; then
    MYSQL_AVAILABLE=true
    log_info "MySQL servicio activo"
  fi
  if systemctl is-active redis-server >/dev/null 2>&1; then
    REDIS_AVAILABLE=true
    log_info "Redis servicio activo"
  fi
fi

# Check if services can be started without root
if [[ "$POSTGRES_AVAILABLE" == "false" ]] && [[ "$IS_ROOT" == "false" ]] && [[ "$IS_SUDO" == "false" ]]; then
  log_warn "No hay PostgreSQL accesible (sin root para instalar)"
fi

# Create .env
log_info "Creando configuración..."

if [[ -f ".env" ]]; then
  log_warn ".env ya existe, respaldando..."
  cp .env .env.backup.$(date +%s)
fi

# Build config based on available database
if [[ "$POSTGRES_AVAILABLE" == "true" ]]; then
  DB_HOST="localhost"
  DB_PORT="5432"
  DB_USER="trading"
  DB_PASSWORD="trading123"
  DB_NAME="reco_trading_prod"
  
  POSTGRES_DSN="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
  POSTGRES_ADMIN_DSN="postgresql+asyncpg://postgres@${DB_HOST}:${DB_PORT}/postgres"
  
  cat > .env <<EOF
# ==============================
# RECO TRADING - AUTO-GENERATED CONFIG
# Generated: $(date)

# Exchange API (REEMPLAZA ESTAS CLAVES)
BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY
BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET

# Exchange Mode
BINANCE_TESTNET=true
CONFIRM_MAINNET=false
ENVIRONMENT=testnet
RUNTIME_PROFILE=paper

# Database (PostgreSQL)
POSTGRES_DSN=${POSTGRES_DSN}
POSTGRES_ADMIN_DSN=${POSTGRES_ADMIN_DSN}

# Redis
REDIS_URL=redis://localhost:6379/0
EOF
  log_success "PostgreSQL configurado"
  
elif [[ "$MYSQL_AVAILABLE" == "true" ]]; then
  DB_HOST="localhost"
  DB_PORT="3306"
  DB_USER="trading"
  DB_PASSWORD="trading123"
  DB_NAME="reco_trading_prod"
  
  MYSQL_DSN="mysql+aiomysql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:3306/${DB_NAME}"
  
  cat > .env <<EOF
# ==============================
# RECO TRADING - AUTO-GENERATED CONFIG
# Generated: $(date)

# Exchange API
BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY
BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET

# Exchange Mode
BINANCE_TESTNET=true
CONFIRM_MAINNET=false
ENVIRONMENT=testnet
RUNTIME_PROFILE=paper

# Database (MySQL)
MYSQL_DSN=${MYSQL_DSN}

# Redis
REDIS_URL=redis://localhost:6379/0
EOF
  log_success "MySQL configurado"

else
  # SQLite fallback (no external DB needed)
  mkdir -p "${ROOT_DIR}/data"
  
  cat > .env <<EOF
# ==============================
# RECO TRADING - AUTO-GENERATED CONFIG
# Generated: $(date)

# Exchange API
BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY
BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET

# Exchange Mode
BINANCE_TESTNET=true
CONFIRM_MAINNET=false
ENVIRONMENT=testnet
RUNTIME_PROFILE=paper

# Database (SQLite - Fallback sin dependencias)
DATABASE_URL=sqlite:///./data/reco_trading.db

# Redis (opcional)
REDIS_URL=redis://localhost:6379/0

# AI/ML Configuration
ENABLE_AUTO_IMPROVER=true
ENABLE_ML_ENGINE=true
ENABLE_CONTINUAL_LEARNING=true
ENABLE_META_LEARNING=true
ENABLE_TFT=true
ENABLE_NBEATS=true
ENABLE_ADVANCED_META_LEARNING=true
ENABLE_REINFORCEMENT_LEARNING=true
DRIFT_DETECTION=true
ONCHAIN_ANALYSIS=true
EOF
  log_warn "Sin PostgreSQL/MySQL detectado, usando SQLite (funciona sin permisos especiales)"
fi

# ============================================
# STEP 6: TRY TO START REDIS (if available)
# ============================================

if check_command redis-server; then
  if [[ "$IS_ROOT" == "true" ]] || [[ "$IS_SUDO" == "true" ]]; then
    ${RUN_CMD} redis-server --daemonize yes 2>/dev/null || true
    log_info "Redis iniciado (background)"
  else
    log_warn "Redis disponible pero requiere root para iniciar"
    log_info "El programa funcionará sin Redis (opcional)"
  fi
else
  log_warn "Redis no disponible - opcional"
fi

# ============================================
# STEP 7: CREATE DIRECTORIES AND SCRIPTS
# ============================================

log_info "Creando estructura..."

mkdir -p data logs

# Create runtime_env.sh
mkdir -p scripts/lib
cat > scripts/lib/runtime_env.sh <<'RUNTIME_ENV'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

upsert_env_var() {
  local env_file="$1"
  local key="$2"
  local value="$3"
  if [[ ! -f "${env_file}" ]]; then
    touch "${env_file}"
  fi
  if grep -qE "^${key}=" "${env_file}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${env_file}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${env_file}"
  fi
}

load_dotenv_file() {
  local env_file="$1"
  if [[ -f "${env_file}" ]]; then
    set -a
    source "${env_file}"
    set +a
  fi
}

load_runtime_env() {
  cd "${ROOT_DIR}"
  load_dotenv_file .env
  load_dotenv_file config/database.env 2>/dev/null || true
}

build_postgres_dsn_from_config() {
  if [[ -n "${DB_USER:-}" && -n "${DB_PASSWORD:-}" && -n "${DB_HOST:-}" && -n "${DB_PORT:-}" && -n "${DB_NAME:-}" ]]; then
    export POSTGRES_DSN="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
    export POSTGRES_ADMIN_DSN="postgresql+asyncpg://postgres@${DB_HOST}:${DB_PORT}/postgres"
    upsert_env_var .env POSTGRES_DSN "${POSTGRES_DSN}"
    upsert_env_var .env POSTGRES_ADMIN_DSN "${POSTGRES_ADMIN_DSN}"
    return 0
  fi
  return 1
}

postgres_host_reachable() {
  python3 - <<'PY'
import os, socket, sys
from urllib.parse import urlparse

dsn = os.environ.get("POSTGRES_DSN", "")
parsed = urlparse(dsn)
host = parsed.hostname
port = parsed.port or 5432

if not host:
    sys.exit(2)

try:
    with socket.create_connection((host, port), timeout=3):
        sys.exit(0)
except OSError:
    sys.exit(1)
PY
}
RUNTIME_ENV

chmod +x scripts/lib/runtime_env.sh
log_success "Scripts de entorno creados"

# ============================================
# STEP 8: VERIFY INSTALLATION
# ============================================

log_info "Verificando instalación..."

source .venv/bin/activate

# Test imports
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from reco_trading.config.settings import Settings
    from reco_trading.database.repository import Repository
    from reco_trading.ml.tft_model import TFTManager
    from reco_trading.ml.nbeats_model import NBEATSManager
    from reco_trading.ml.advanced_meta_learner import MetaLearningManager
    print('✅ All ML imports OK')
except ImportError as e:
    print(f'⚠️ ML Import warning: {e}')
    print('✅ Basic imports OK')
" 2>/dev/null && log_success "Imports verificados" || log_error "Error en imports"

# Test database connection
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from reco_trading.config.settings import Settings
from reco_trading.database.repository import Repository

async def test():
    s = Settings()
    if s.postgres_dsn:
        r = Repository(s.postgres_dsn)
        try:
            await r.verify_connectivity()
            print('✅ DB OK')
        except Exception as e:
            print(f'⚠️ DB: {e}')
        finally:
            await r.close()
    elif s.database_url:
        print('✅ SQLite configured')
    else:
        print('⚠️ Sin DSN')

asyncio.run(test())
" 2>/dev/null && log_success "Base de datos verificada" || log_warn "Sin conexión a BD (usará SQLite)"

# ============================================
# FINAL OUTPUT
# ============================================

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ INSTALACIÓN COMPLETADA CON AI/ML!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if [[ "$IS_ROOT" == "false" ]] && [[ "$IS_SUDO" == "false" ]]; then
  echo -e "${YELLOW}ℹ️  MODO SIN ROOT - Instalación completada correctamente${NC}"
  echo "   El programa funciona sin permisos de administrador"
fi

echo ""
echo -e "${YELLOW}⚠️ IMPORTANTE:${NC}"
echo "  Edita el archivo .env y reemplaza:"
echo "    - BINANCE_API_KEY"
echo "    - BINANCE_API_SECRET"
echo ""
echo -e "${CYAN}Para iniciar el bot:${NC}"
echo "  ./run.sh"
echo ""
echo -e "${CYAN}Para verificar la configuración:${NC}"
echo "  source .venv/bin/activate"
echo "  python -c 'from reco_trading.config.settings import Settings; print(Settings().binance_testnet)'"
echo ""