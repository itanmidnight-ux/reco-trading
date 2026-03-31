#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# ============================================
# DETECT ROOT ACCESS
# ============================================

IS_ROOT=false
IS_SUDO=false

if [[ "${EUID}" -eq 0 ]]; then
  IS_ROOT=true
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  IS_SUDO=true
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  RECO-TRADING BOT v3.0${NC}"
echo -e "${CYAN}========================================${NC}"

if [[ "$IS_ROOT" == "true" ]]; then
  log_info "Ejecutando como ROOT"
elif [[ "$IS_SUDO" == "true" ]]; then
  log_info "SUDO disponible"
else
  log_info "Modo usuario (sin root)"
fi

# Load functions (handle missing file gracefully)
if [[ -f "${ROOT_DIR}/scripts/lib/runtime_env.sh" ]]; then
  source "${ROOT_DIR}/scripts/lib/runtime_env.sh"
else
  load_runtime_env() {
    if [[ -f "${ROOT_DIR}/.env" ]]; then
      set -a
      source "${ROOT_DIR}/.env"
      set +a
    fi
  }
fi

# ============================================
# CHECK VIRTUAL ENVIRONMENT
# ============================================

if [[ ! -d "${ROOT_DIR}/.venv" ]]; then
  log_error "Entorno virtual no encontrado"
  echo ""
  echo "Ejecuta install.sh primero:"
  echo "  ./install.sh"
  exit 1
fi

# Check if already in venv
IN_VENV=$(${ROOT_DIR}/.venv/bin/python -c "import sys; print('yes' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'no')" 2>/dev/null || echo "no")

if [[ "$IN_VENV" != "yes" ]]; then
  log_info "Activando entorno virtual..."
  source "${ROOT_DIR}/.venv/bin/activate"
fi

# Verify Python environment
PYTHON_BIN=$(which python)
echo -e "${GREEN}Python: $($PYTHON_BIN --version 2>&1)${NC}"

# ============================================
# VERIFY DEPENDENCIES
# ============================================

log_info "Verificando dependencias..."

MISSING_DEPS=()
for pkg in ccxt asyncpg sqlalchemy httpx; do
  if ! $PYTHON_BIN -c "import $pkg" 2>/dev/null; then
    MISSING_DEPS+=("$pkg")
  fi
done

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
  log_error "Faltan dependencias: ${MISSING_DEPS[*]}"
  echo ""
  echo "Ejecuta:"
  echo "  ./install.sh"
  echo ""
  echo "O instala manualmente:"
  echo "  pip install ${MISSING_DEPS[*]}"
  exit 1
fi

echo -e "${GREEN}✓ Dependencias OK${NC}"

# ============================================
# LOAD ENVIRONMENT
# ============================================

load_runtime_env

# ============================================
# VERIFY API KEYS
# ============================================

if [[ -z "${BINANCE_API_KEY:-}" ]] || [[ "$BINANCE_API_KEY" == "CAMBIAR_POR_TU_API_KEY" ]]; then
  log_error "BINANCE_API_KEY no configurada en .env"
  echo ""
  echo "Edita el archivo .env y reemplaza:"
  echo "  BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY"
  echo ""
  echo "Por tus credenciales de Binance."
  exit 1
fi

if [[ -z "${BINANCE_API_SECRET:-}" ]] || [[ "$BINANCE_API_SECRET" == "CAMBIAR_POR_TU_API_SECRET" ]]; then
  log_error "BINANCE_API_SECRET no configurada en .env"
  echo ""
  echo "Edita el archivo .env y reemplaza:"
  echo "  BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET"
  echo ""
  echo "Por tus credenciales de Binance."
  exit 1
fi

echo -e "${GREEN}✓ API Keys configuradas${NC}"

# ============================================
# MENU - SELECT MODE
# ============================================

echo ""
echo "Seleccione modo de ejecución:"
echo "1) Testnet (Sandbox) - Recomendado para pruebas"
echo "2) Producción Real (Dinero real)"
read -p "Opción [1]: " MODE_OPTION

if [[ "$MODE_OPTION" == "2" ]]; then
  read -p "Escriba 'CONFIRMAR' para operar con dinero real: " CONFIRM
  if [[ "$CONFIRM" != "CONFIRMAR" ]]; then
    echo "Cancelado."
    exit 1
  fi
  export BINANCE_TESTNET=false
  export CONFIRM_MAINNET=true
  export ENVIRONMENT=production
  export RUNTIME_PROFILE=live
  echo -e "${RED}⚠️  MODO PRODUCCIÓN ACTIVADO${NC}"
else
  export BINANCE_TESTNET=true
  export CONFIRM_MAINNET=false
  export ENVIRONMENT=testnet
  export RUNTIME_PROFILE=paper
  echo -e "${GREEN}✓ MODO TESTNET${NC}"
fi

# Update .env with mode
if [[ -f "${ROOT_DIR}/.env" ]]; then
  sed -i "s|^BINANCE_TESTNET=.*|BINANCE_TESTNET=${BINANCE_TESTNET}|" "${ROOT_DIR}/.env" 2>/dev/null || true
  sed -i "s|^CONFIRM_MAINNET=.*|CONFIRM_MAINNET=${CONFIRM_MAINNET}|" "${ROOT_DIR}/.env" 2>/dev/null || true
  sed -i "s|^ENVIRONMENT=.*|ENVIRONMENT=${ENVIRONMENT}|" "${ROOT_DIR}/.env" 2>/dev/null || true
fi

# ============================================
# CHECK DATABASE
# ============================================

log_info "Verificando base de datos..."

DB_STATUS="unknown"

# Try PostgreSQL
if [[ -n "${POSTGRES_DSN:-}" ]]; then
  if $PYTHON_BIN -c "
import asyncio
import sys
sys.path.insert(0, '.')
from reco_trading.database.repository import Repository

async def check():
    from reco_trading.config.settings import Settings
    s = Settings()
    if s.postgres_dsn:
        r = Repository(s.postgres_dsn)
        try:
            await r.verify_connectivity()
            print('OK')
        except:
            pass
        finally:
            await r.close()

asyncio.run(check())
" 2>/dev/null; then
    DB_STATUS="postgresql"
    echo -e "${GREEN}✓ PostgreSQL conectado${NC}"
  else
    log_warn "PostgreSQL no accesible"
  fi
fi

# Try MySQL
if [[ -n "${MYSQL_DSN:-}" ]] && [[ "$DB_STATUS" == "unknown" ]]; then
  if $PYTHON_BIN -c "
import asyncio
import sys
sys.path.insert(0, '.')
from reco_trading.database.repository import Repository

async def check():
    from reco_trading.config.settings import Settings
    s = Settings()
    if s.mysql_dsn:
        r = Repository(s.mysql_dsn)
        try:
            await r.verify_connectivity()
            print('OK')
        except:
            pass
        finally:
            await r.close()

asyncio.run(check())
" 2>/dev/null; then
    DB_STATUS="mysql"
    echo -e "${GREEN}✓ MySQL conectado${NC}"
  fi
fi

# Fallback to SQLite
if [[ "$DB_STATUS" == "unknown" ]]; then
  if [[ -n "${DATABASE_URL:-}" ]]; then
    DB_STATUS="sqlite"
    echo -e "${GREEN}✓ SQLite configurado${NC}"
  else
    log_warn "Sin base de datos configurada"
    echo "  El programa usará SQLite automáticamente"
    DB_STATUS="sqlite_fallback"
  fi
fi

# ============================================
# CLEAN UP OLD PROCESSES
# ============================================

if pgrep -f "python.*reco_trading" > /dev/null 2>&1; then
  log_warn "Deteniendo instancia anterior..."
  pkill -f "python.*reco_trading" 2>/dev/null || true
  sleep 2
fi

# Free ports if we have permissions
for port in 8000 8080 9000; do
  if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
    if [[ "$IS_ROOT" == "true" ]] || [[ "$IS_SUDO" == "true" ]]; then
      log_info "Liberando puerto ${port}..."
      fuser -k ${port}/tcp 2>/dev/null || true
      sleep 1
    fi
  fi
done

# ============================================
# START BOT
# ============================================

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ INICIANDO BOT${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Modo: ${ENVIRONMENT}"
echo "Base de datos: ${DB_STATUS}"
echo ""

# Run the bot
python main.py "$@"