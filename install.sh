#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Reco-Trading Installer${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/lib/runtime_env.sh"

if [[ "${EUID}" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

ensure_env_file() {
  local env_file=".env"

  if [[ -f "${env_file}" ]]; then
    echo -e "${YELLOW}Actualizando archivo ${env_file} con configuración base...${NC}"
  else
    echo -e "${GREEN}Creando archivo ${env_file} con configuración base...${NC}"
    cat > "${env_file}" <<'ENV_TEMPLATE'
# ==============================
# RECO TRADING - CONFIGURACIÓN
# ==============================
# NOTA: Reemplaza manualmente BINANCE_API_KEY y BINANCE_API_SECRET.
# El script run.sh actualiza automáticamente BINANCE_TESTNET,
# CONFIRM_MAINNET, ENVIRONMENT y RUNTIME_PROFILE según el modo elegido.

BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY
BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET

# Modo por defecto (run.sh lo ajusta dinámicamente)
BINANCE_TESTNET=true
CONFIRM_MAINNET=false
ENVIRONMENT=testnet
RUNTIME_PROFILE=paper

# Infraestructura
POSTGRES_DSN=
POSTGRES_ADMIN_DSN=
REDIS_URL=redis://localhost:6379/0
ENV_TEMPLATE
  fi

  upsert_env_var "${env_file}" "BINANCE_TESTNET" "true"
  upsert_env_var "${env_file}" "CONFIRM_MAINNET" "false"
  upsert_env_var "${env_file}" "ENVIRONMENT" "testnet"
  upsert_env_var "${env_file}" "RUNTIME_PROFILE" "paper"
  upsert_env_var "${env_file}" "REDIS_URL" "redis://localhost:6379/0"
}

echo -e "${BLUE}>>> Instalando dependencias del sistema...${NC}"
export DEBIAN_FRONTEND=noninteractive
${SUDO} apt-get update
${SUDO} apt-get install -y python3-venv redis-server

echo -e "${BLUE}>>> Configurando entorno virtual...${NC}"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${BLUE}>>> Asegurando servicios auxiliares...${NC}"
if command -v systemctl >/dev/null 2>&1; then
  ${SUDO} systemctl enable redis-server || true
  ${SUDO} systemctl start redis-server || true
  ${SUDO} timedatectl set-ntp true || true
  ${SUDO} systemctl restart systemd-timesyncd || true
  ${SUDO} timedatectl status || true
elif command -v service >/dev/null 2>&1; then
  ${SUDO} service redis-server start || true
fi

ensure_env_file

echo -e "${BLUE}>>> Configurando PostgreSQL...${NC}"
if [[ -f scripts/postgres/bootstrap_local_postgres.sh ]]; then
  chmod +x scripts/postgres/bootstrap_local_postgres.sh
  scripts/postgres/bootstrap_local_postgres.sh
fi

if [[ -f scripts/ensure_postgres.sh ]]; then
  chmod +x scripts/ensure_postgres.sh
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ Instalación completada!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Próximos pasos:${NC}"
echo "  1. Edita el archivo .env con tus credenciales de Binance"
echo "  2. Ejecuta ./run.sh para iniciar el bot"
echo ""
echo -e "${BLUE}Para más información, consulta README.md${NC}"
