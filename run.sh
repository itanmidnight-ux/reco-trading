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

  echo "Intentando corregir PostgreSQL automáticamente..."
  if ! "${helper}"; then
    return 1
  fi

  load_runtime_env
  if [ -z "${POSTGRES_DSN:-}" ]; then
    build_postgres_dsn_from_config || true
  fi
}

# Activar entorno virtual si existe
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "Aviso: .venv/bin/activate no existe. Usando Python del sistema."
fi

load_runtime_env

# Intentar construir DSN desde config si no existe
if [ -z "${POSTGRES_DSN:-}" ]; then
  build_postgres_dsn_from_config || true
fi

# Menú de selección de modo
echo ""
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

echo ""

# Verificar API Keys
missing_vars=()
for required_var in BINANCE_API_KEY BINANCE_API_SECRET; do
  if [ -z "${!required_var:-}" ]; then
    missing_vars+=("${required_var}")
  fi
done

# Verificar base de datos - preferir PostgreSQL, fallback a SQLite
db_available=false

# Intentar PostgreSQL primero
if [ -n "${POSTGRES_DSN:-}" ]; then
  if postgres_host_reachable; then
    echo "✓ PostgreSQL conectado correctamente"
    db_available=true
  else
    echo "Advertencia: PostgreSQL no está disponible en el host configurado"
    # Intentar auto-reparar
    if attempt_postgres_auto_fix && postgres_host_reachable; then
      echo "✓ PostgreSQL fue reparado automáticamente"
      db_available=true
    fi
  fi
fi

# Fallback a SQLite si PostgreSQL no está disponible
if [ "$db_available" = false ]; then
  echo "Usando SQLite como fallback..."
  build_sqlite_dsn
  if [ -n "${DATABASE_URL:-}" ]; then
    echo "✓ SQLite configurado: ${DATABASE_URL}"
    db_available=true
  fi
fi

if [ "$db_available" = false ]; then
  echo "Error: No se pudo configurar ninguna base de datos"
  exit 1
fi

echo ""
echo "Iniciando Reco-Trading Bot..."
echo ""

python main.py