#!/usr/bin/env bash
set -euo pipefail

upsert_env_var() {
  local env_file="$1"
  local key="$2"
  local value="$3"

  if grep -qE "^${key}=" "${env_file}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${env_file}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${env_file}"
  fi
}

sync_mode_to_env() {
  local mode_option="$1"
  local env_file=".env"

  if [ ! -f "${env_file}" ]; then
    touch "${env_file}"
  fi

  if [ "${mode_option}" = "1" ]; then
    upsert_env_var "${env_file}" "BINANCE_TESTNET" "true"
    upsert_env_var "${env_file}" "CONFIRM_MAINNET" "false"
    upsert_env_var "${env_file}" "ENVIRONMENT" "testnet"
    upsert_env_var "${env_file}" "RUNTIME_PROFILE" "paper"
  elif [ "${mode_option}" = "2" ]; then
    upsert_env_var "${env_file}" "BINANCE_TESTNET" "false"
    upsert_env_var "${env_file}" "CONFIRM_MAINNET" "true"
    upsert_env_var "${env_file}" "ENVIRONMENT" "production"
    upsert_env_var "${env_file}" "RUNTIME_PROFILE" "production"
  fi
}

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "Aviso: .venv/bin/activate no existe. Usando Python del sistema."
fi

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -z "${POSTGRES_DSN:-}" ] && [ -f config/database.env ]; then
  set -a
  # shellcheck disable=SC1091
  source config/database.env
  set +a
  if [ -n "${DB_USER:-}" ] && [ -n "${DB_PASSWORD:-}" ] && [ -n "${DB_HOST:-}" ] && [ -n "${DB_PORT:-}" ] && [ -n "${DB_NAME:-}" ]; then
    export POSTGRES_DSN="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
    upsert_env_var .env POSTGRES_DSN "${POSTGRES_DSN}"
  fi
fi

echo "Seleccione modo de ejecución:"
echo "1) Binance Testnet (Sandbox - Simulación)"
echo "2) Binance Producción Real (Mainnet - Dinero real)"
read -r -p "Ingrese opción (1 o 2): " MODE_OPTION

if [ "$MODE_OPTION" = "1" ]; then
  export BINANCE_TESTNET=true
  export CONFIRM_MAINNET=false
  export ENVIRONMENT=testnet
  export RUNTIME_PROFILE=paper
  sync_mode_to_env "$MODE_OPTION"
  echo "Modo TESTNET activado."
elif [ "$MODE_OPTION" = "2" ]; then
  export BINANCE_TESTNET=false
  read -r -p "⚠️  Está a punto de operar con dinero real. Escriba CONFIRMAR para continuar: " CONFIRM
  if [ "$CONFIRM" != "CONFIRMAR" ]; then
    echo "Operación cancelada."
    exit 1
  fi
  if [ -z "${BINANCE_API_KEY:-}" ] || [ -z "${BINANCE_API_SECRET:-}" ]; then
    echo "Error: BINANCE_API_KEY y BINANCE_API_SECRET son obligatorios para producción real."
    exit 1
  fi
  export CONFIRM_MAINNET=true
  export ENVIRONMENT=production
  export RUNTIME_PROFILE=production
  sync_mode_to_env "$MODE_OPTION"
  echo "Modo PRODUCCIÓN REAL activado."
else
  echo "Opción inválida."
  exit 1
fi

missing_vars=()
for required_var in BINANCE_API_KEY BINANCE_API_SECRET POSTGRES_DSN; do
  if [ -z "${!required_var:-}" ]; then
    missing_vars+=("${required_var}")
  fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
  echo "Error: faltan variables obligatorias: ${missing_vars[*]}"
  echo "Sugerencia rápida:"
  echo "  1) Ejecuta ./install.sh para preparar PostgreSQL y sincronizar el DSN."
  echo "  2) Edita .env con tus credenciales de Binance."
  echo "     Variables requeridas en .env:"
  echo "       BINANCE_API_KEY=tu_api_key"
  echo "       BINANCE_API_SECRET=tu_api_secret"
  echo "       POSTGRES_DSN=postgresql+asyncpg://trading:***@localhost:5432/reco_trading_prod"
  exit 1
fi

DASHBOARD_HOST="${DASHBOARD_HOST:-127.0.0.1}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"
AUTO_START_WEB="${AUTO_START_WEB:-true}"

cleanup() {
  if [ -n "${WEB_PID:-}" ] && kill -0 "$WEB_PID" 2>/dev/null; then
    kill "$WEB_PID" 2>/dev/null || true
    wait "$WEB_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [ "$AUTO_START_WEB" = "true" ]; then
  echo "Iniciando dashboard web en http://${DASHBOARD_HOST}:${DASHBOARD_PORT} ..."
  python -m uvicorn reco_trading.web.dashboard:app --host "$DASHBOARD_HOST" --port "$DASHBOARD_PORT" --log-level warning &
  WEB_PID=$!
  sleep 1
  if ! kill -0 "$WEB_PID" 2>/dev/null; then
    echo "Error: no se pudo iniciar el dashboard web."
    exit 1
  fi
fi

python main.py
