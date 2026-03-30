#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Cargar funciones
source "${ROOT_DIR}/scripts/lib/runtime_env.sh" 2>/dev/null || true

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  RECO-TRADING BOT${NC}"
echo -e "${BLUE}========================================${NC}"

# Verificar si ya hay una instancia corriendo
if pgrep -f "python.*main.py" > /dev/null 2>&1; then
  echo -e "${YELLOW}Ya hay una instancia del bot corriendo.${NC}"
  echo "Deteniendo instancia anterior..."
  pkill -f "python.*main.py" || true
  sleep 2
fi

# Liberar puerto 8000 si está en uso
if ss -tlnp 2>/dev/null | grep -q ":8000 "; then
  echo -e "${YELLOW}Puerto 8000 en uso, liberando...${NC}"
  fuser -k 8000/tcp 2>/dev/null || true
  sleep 1
fi

# Activar entorno virtual
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  echo -e "${YELLOW}Advertencia: .venv no encontrado${NC}"
fi

# Cargar variables de entorno
load_runtime_env 2>/dev/null || true

# Menú de modo
echo ""
echo "Seleccione modo de ejecución:"
echo "1) Testnet (Sandbox)"
echo "2) Producción Real (Dinero real)"
read -p "Opción [1]: " MODE_OPTION

if [ "$MODE_OPTION" = "2" ]; then
  read -p "Escriba 'CONFIRMAR' para operar con dinero real: " CONFIRM
  if [ "$CONFIRM" != "CONFIRMAR" ]; then
    echo "Cancelado."
    exit 1
  fi
  export BINANCE_TESTNET=false
  export CONFIRM_MAINNET=true
  export ENVIRONMENT=production
  echo -e "${RED}Modo PRODUCCIÓN${NC}"
else
  export BINANCE_TESTNET=true
  export CONFIRM_MAINNET=false
  export ENVIRONMENT=testnet
  echo -e "${GREEN}Modo TESTNET${NC}"
fi

# Verificar variables requeridas
if [ -z "${BINANCE_API_KEY:-}" ] || [ -z "${BINANCE_API_SECRET:-}" ]; then
  echo -e "${RED}Error: BINANCE_API_KEY y BINANCE_API_SECRET son requeridos${NC}"
  echo "Configúralos en el archivo .env"
  exit 1
fi

# Verificar PostgreSQL
if [ -z "${POSTGRES_DSN:-}" ]; then
  echo -e "${YELLOW}PostgreSQL DSN no configurado, usando SQLite${NC}"
fi

echo ""
echo -e "${GREEN}Iniciando bot con Dashboard...${NC}"
echo ""

python main.py
