#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Colores

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Reco-Trading Installer (Termux + PG)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Sudo

if [[ "${EUID}" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
SUDO="sudo"
else
SUDO=""
fi

# ==============================

# ENV helper

# ==============================

upsert_env_var() {
local file="$1"
local key="$2"
local value="$3"

if grep -q "^${key}=" "$file" 2>/dev/null; then
sed -i "s|^${key}=.*|${key}=${value}|" "$file"
else
echo "${key}=${value}" >> "$file"
fi
}

ensure_env_file() {
local env_file=".env"

if [[ -f "${env_file}" ]]; then
echo -e "${YELLOW}Actualizando ${env_file}...${NC}"
else
echo -e "${GREEN}Creando ${env_file}...${NC}"
cat > "${env_file}" <<'EOF'
BINANCE_API_KEY=CAMBIAR
BINANCE_API_SECRET=CAMBIAR

BINANCE_TESTNET=true
CONFIRM_MAINNET=false
ENVIRONMENT=testnet
RUNTIME_PROFILE=paper

POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/reco
POSTGRES_ADMIN_DSN=postgresql://postgres:postgres@localhost:5432/postgres
REDIS_URL=redis://localhost:6379/0
EOF
fi
}

echo -e "${BLUE}>>> Instalando dependencias...${NC}"
export DEBIAN_FRONTEND=noninteractive
${SUDO} apt-get update
${SUDO} apt-get install -y python3-venv redis-server postgresql postgresql-contrib

echo -e "${BLUE}>>> Configurando entorno Python...${NC}"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# ==============================

# REDIS

# ==============================

echo -e "${BLUE}>>> Iniciando Redis...${NC}"
redis-server --daemonize yes || true

# ==============================

# POSTGRESQL (modo proot)

# ==============================

echo -e "${BLUE}>>> Configurando PostgreSQL (modo manual)...${NC}"

PGDATA="$HOME/pgdata"

if [[ ! -d "$PGDATA" ]]; then
echo -e "${YELLOW}Inicializando base de datos...${NC}"
initdb -D "$PGDATA"
fi

echo -e "${BLUE}Iniciando PostgreSQL...${NC}"
pg_ctl -D "$PGDATA" -l logfile start || true

# Crear DB si no existe

echo -e "${BLUE}Creando base de datos reco...${NC}"
createdb reco || true

ensure_env_file

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ Instalación completa con PostgreSQL${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}IMPORTANTE:${NC}"
echo "Cada vez que entres a Termux ejecuta:"
echo ""
echo "  pg_ctl -D $HOME/pgdata -l logfile start"
echo "  redis-server --daemonize yes"
echo ""
echo "Luego:"
echo "  ./run.sh"
