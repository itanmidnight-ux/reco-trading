#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/lib/runtime_env.sh"

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

attempt_postgres_auto_fix() {
  local helper="scripts/postgres/bootstrap_local_postgres.sh"
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

  load_runtime_env
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

load_runtime_env

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
  if attempt_postgres_auto_fix && postgres_host_reachable; then
    echo "PostgreSQL fue reparado automáticamente. Continuando con el arranque..."
  else
    echo "Error: PostgreSQL no está disponible en el host/puerto configurados por POSTGRES_DSN."
    echo "Sugerencia rápida:"
    echo "  1) Ejecuta scripts/postgres/bootstrap_local_postgres.sh o corrige DB_HOST/DB_PORT en config/database.env o POSTGRES_DSN en .env."
    echo "  2) Verifica conectividad antes de arrancar el bot."
    echo "     Ejemplo esperado: postgresql+asyncpg://usuario:clave@localhost:5432/reco_trading_prod"
    exit 1
  fi
fi

python main.py
