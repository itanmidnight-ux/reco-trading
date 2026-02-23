#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

MODE="${RUN_MODE:-testnet}" # testnet|mainnet
AUTO_INSTALL="${AUTO_INSTALL:-true}" # true|false
PRECHECK_ONLY="${PRECHECK_ONLY:-false}" # true|false
PY_BIN="python3"

upsert_env_var() {
  local env_file="$1"
  local key="$2"
  local value="$3"

  if [ ! -f "${env_file}" ]; then
    touch "${env_file}"
  fi

  if grep -qE "^${key}=" "${env_file}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${env_file}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${env_file}"
  fi
}

ensure_venv() {
  if [ ! -f .venv/bin/activate ]; then
    echo "[run] Entorno virtual no encontrado. Creando .venv..."
    ${PY_BIN} -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate

  if ! python -c 'import fastapi, pydantic, ccxt' >/dev/null 2>&1; then
    echo "[run] Dependencias incompletas. Instalando requirements.txt..."
    pip install --upgrade pip >/dev/null
    pip install -r requirements.txt >/dev/null
  fi
}

load_env() {
  if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
}

sync_database_dsn() {
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

  if [ -z "${REDIS_URL:-}" ]; then
    export REDIS_URL="redis://localhost:6379/0"
    upsert_env_var .env REDIS_URL "${REDIS_URL}"
  fi
}

configure_mode() {
  case "${MODE}" in
    testnet)
      export BINANCE_TESTNET=true
      export CONFIRM_MAINNET=false
      export ENVIRONMENT=testnet
      export RUNTIME_PROFILE=paper
      ;;
    mainnet)
      export BINANCE_TESTNET=false
      export CONFIRM_MAINNET=true
      export ENVIRONMENT=production
      export RUNTIME_PROFILE=production
      ;;
    *)
      echo "[run] RUN_MODE inválido: ${MODE}. Use testnet o mainnet."
      exit 1
      ;;
  esac

  upsert_env_var .env BINANCE_TESTNET "${BINANCE_TESTNET}"
  upsert_env_var .env CONFIRM_MAINNET "${CONFIRM_MAINNET}"
  upsert_env_var .env ENVIRONMENT "${ENVIRONMENT}"
  upsert_env_var .env RUNTIME_PROFILE "${RUNTIME_PROFILE}"
}

validate_keys_not_placeholder() {
  if [ "${BINANCE_API_KEY:-}" = "CAMBIAR_POR_TU_API_KEY" ] || [ "${BINANCE_API_SECRET:-}" = "CAMBIAR_POR_TU_API_SECRET" ]; then
    echo "[run] BINANCE_API_KEY/BINANCE_API_SECRET contienen placeholders. Edita .env antes de ejecutar."
    exit 1
  fi
}

run_preflight() {
  local mode_arg="testnet"
  if [ "${MODE}" = "mainnet" ]; then
    mode_arg="mainnet"
  fi

  if ! python -m reco_trading.system.preflight --mode "${mode_arg}"; then
    if [ "${AUTO_INSTALL}" = "true" ]; then
      echo "[run] Preflight falló. Intentando reparación automática con ./install.sh ..."
      ./install.sh
      load_env
      sync_database_dsn
      python -m reco_trading.system.preflight --mode "${mode_arg}"
    else
      echo "[run] Preflight falló y AUTO_INSTALL=false."
      exit 1
    fi
  fi
}

start_runtime() {
  echo "[run] Iniciando runtime: mode=${MODE}, environment=${ENVIRONMENT}, profile=${RUNTIME_PROFILE}"
  exec python main.py --env "${MODE}" --mode live
}

ensure_venv
load_env
sync_database_dsn
configure_mode
validate_keys_not_placeholder
run_preflight

if [ "${PRECHECK_ONLY}" = "true" ]; then
  echo "[run] PRECHECK_ONLY=true, finalizando sin iniciar runtime."
  exit 0
fi

start_runtime
