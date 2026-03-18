#!/usr/bin/env bash
set -euo pipefail

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

load_database_config() {
  if [ -f config/database.env ]; then
    set -a
    # shellcheck disable=SC1091
    source config/database.env
    set +a
  fi
}

build_postgres_dsn_from_config() {
  if [ -n "${DB_USER:-}" ] && [ -n "${DB_PASSWORD:-}" ] && [ -n "${DB_HOST:-}" ] && [ -n "${DB_PORT:-}" ] && [ -n "${DB_NAME:-}" ]; then
    export POSTGRES_DSN="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
    upsert_env_var .env POSTGRES_DSN "${POSTGRES_DSN}"
    return 0
  fi
  return 1
}

postgres_host_reachable() {
  python - <<'PY'
from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

dsn = os.environ.get("POSTGRES_DSN", "")
parsed = urlparse(dsn)
host = parsed.hostname
port = parsed.port or 5432

if not host:
    raise SystemExit(2)

try:
    with socket.create_connection((host, port), timeout=3):
        raise SystemExit(0)
except OSError:
    raise SystemExit(1)
PY
}

attempt_postgres_auto_fix() {
  local helper="scripts/ensure_postgres.sh"
  case "${DB_HOST:-localhost}" in
    localhost|127.0.0.1|::1) ;;
    *)
      return 1
      ;;
  esac

  if [ ! -f "${helper}" ]; then
    return 1
  fi

  if [ ! -x "${helper}" ]; then
    chmod +x "${helper}"
  fi

  echo "Detecté un problema con PostgreSQL. Intentando corregirlo automáticamente..."
  if ! "${helper}"; then
    return 1
  fi

  if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
  load_database_config
  if [ -z "${POSTGRES_DSN:-}" ]; then
    build_postgres_dsn_from_config || true
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

load_database_config

if [ -z "${POSTGRES_DSN:-}" ]; then
  build_postgres_dsn_from_config || true
fi

echo "Seleccione modo de ejecución:"
echo "1) Binance Testnet (Sandbox - ÓRDENES REALES EN TESTNET)"
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

if [ "${#missing_vars[@]}" -gt 0 ] && printf '%s\n' "${missing_vars[@]}" | grep -qx "POSTGRES_DSN"; then
  if attempt_postgres_auto_fix; then
    missing_vars=()
    for required_var in BINANCE_API_KEY BINANCE_API_SECRET POSTGRES_DSN; do
      if [ -z "${!required_var:-}" ]; then
        missing_vars+=("${required_var}")
      fi
    done
  fi
fi

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

if ! postgres_host_reachable; then
  if ! attempt_postgres_auto_fix || ! postgres_host_reachable; then
    echo "Error: PostgreSQL no está disponible en el host/puerto configurados por POSTGRES_DSN."
    echo "Sugerencia rápida:"
    echo "  1) Revisa config/database.env y asegúrate de que DB_USER, DB_PASSWORD, DB_NAME, DB_HOST y DB_PORT estén completos."
    echo "  2) Ejecuta ./scripts/ensure_postgres.sh o ./install.sh para aprovisionar PostgreSQL."
    echo "     Ejemplo esperado: postgresql+asyncpg://usuario:clave@localhost:5432/reco_trading_prod"
    exit 1
  fi
fi

python main.py
