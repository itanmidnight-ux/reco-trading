#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ENV_LIB="${ROOT_DIR}/scripts/lib/runtime_env.sh"

ensure_runtime_env_lib_exists() {
  if [ -f "${RUNTIME_ENV_LIB}" ]; then
    return 0
  fi

  mkdir -p "$(dirname "${RUNTIME_ENV_LIB}")"
  cat > "${RUNTIME_ENV_LIB}" <<'RUNTIME_LIB'
#!/usr/bin/env bash

load_dotenv_file() {
  local dotenv_file="${1:-.env}"
  if [ -f "${dotenv_file}" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${dotenv_file}"
    set +a
  fi
}

# Definición activa de upsert_env_var (sobreescribe la del runtime_env.sh cargado arriba)
# Usa python3 como método primario para máxima compatibilidad cross-distro
upsert_env_var() {
  local env_file="$1"
  local key="$2"
  local value="$3"
  touch "${env_file}"
  if command -v awk >/dev/null 2>&1; then
    awk -v k="$key" -v v="$value" '
      BEGIN { done=0 }
      $0 ~ ("^" k "=") { print k "=" v; done=1; next }
      { print }
      END { if (done==0) print k "=" v }
    ' "${env_file}" > "${env_file}.tmp" && mv "${env_file}.tmp" "${env_file}"
  else
    if grep -qE "^${key}=" "${env_file}" 2>/dev/null; then
      sed "s|^${key}=.*|${key}=${value}|" "${env_file}" > "${env_file}.tmp"
      mv "${env_file}.tmp" "${env_file}"
    else
      printf '%s=%s\n' "${key}" "${value}" >> "${env_file}"
    fi
  fi
}

build_postgres_dsn_from_config() {
  local pg_user="${POSTGRES_USER:-${DB_USER:-postgres}}"
  local pg_pass="${POSTGRES_PASSWORD:-${DB_PASSWORD:-postgres}}"
  local pg_host="${POSTGRES_HOST:-${DB_HOST:-localhost}}"
  local pg_port="${POSTGRES_PORT:-${DB_PORT:-5432}}"
  local pg_db="${POSTGRES_DB:-${DB_NAME:-reco_trading}}"
  POSTGRES_DSN="postgresql+asyncpg://${pg_user}:${pg_pass}@${pg_host}:${pg_port}/${pg_db}"
  export POSTGRES_DSN
}

build_sqlite_dsn() {
  local data_dir
  data_dir="$(pwd)/data"
  mkdir -p "${data_dir}"
  DATABASE_URL="sqlite:///${data_dir}/reco_trading.db"
  export DATABASE_URL
}
RUNTIME_LIB
  chmod +x "${RUNTIME_ENV_LIB}"
}

ensure_runtime_env_lib_exists
# shellcheck disable=SC1091
source "${RUNTIME_ENV_LIB}"

# Definición activa de upsert_env_var (sobreescribe la del runtime_env.sh cargado arriba)
# Usa python3 como método primario para máxima compatibilidad cross-distro
upsert_env_var() {
  local env_file="$1"
  local key="$2"
  local value="$3"

  mkdir -p "$(dirname "${env_file}")"
  touch "${env_file}"

  # Método 1: python3 (más robusto para cualquier distro Linux).
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$env_file" "$key" "$value" <<'PY'
import sys
from pathlib import Path

env_file, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
path = Path(env_file)
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
needle = f"{key}="
updated = False
out = []
for line in lines:
    if line.startswith(needle):
        out.append(f"{key}={value}")
        updated = True
    else:
        out.append(line)
if not updated:
    out.append(f"{key}={value}")
path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
PY
    return 0
  fi

  # Método 2: awk (fallback universal POSIX).
  if command -v awk >/dev/null 2>&1; then
    awk -v k="$key" -v v="$value" '
      BEGIN { done=0 }
      $0 ~ ("^" k "=") { print k "=" v; done=1; next }
      { print }
      END { if (done==0) print k "=" v }
    ' "${env_file}" > "${env_file}.tmp" && mv "${env_file}.tmp" "${env_file}"
    return 0
  fi

  # Método 3: sed + grep (último recurso).
  if grep -qE "^${key}=" "${env_file}" 2>/dev/null; then
    sed "s|^${key}=.*|${key}=${value}|" "${env_file}" > "${env_file}.tmp"
    mv "${env_file}.tmp" "${env_file}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${env_file}"
  fi
}

load_runtime_env() {
  local env_file=".env"
  if [ -f "${env_file}" ]; then
    # shellcheck disable=SC1090
    set -a
    source "${env_file}"
    set +a
  fi
}

build_postgres_dsn_from_config() {
  local pg_user="${POSTGRES_USER:-${DB_USER:-postgres}}"
  local pg_pass="${POSTGRES_PASSWORD:-${DB_PASSWORD:-postgres}}"
  local pg_host="${POSTGRES_HOST:-${DB_HOST:-localhost}}"
  local pg_port="${POSTGRES_PORT:-${DB_PORT:-5432}}"
  local pg_db="${POSTGRES_DB:-${DB_NAME:-reco_trading}}"

  POSTGRES_DSN="postgresql+asyncpg://${pg_user}:${pg_pass}@${pg_host}:${pg_port}/${pg_db}"
  export POSTGRES_DSN
}

postgres_host_reachable() {
  local host="${DB_HOST:-${POSTGRES_HOST:-localhost}}"
  local port="${DB_PORT:-${POSTGRES_PORT:-5432}}"

  # Método 1: pg_isready (si está disponible).
  if command -v pg_isready >/dev/null 2>&1; then
    pg_isready -h "${host}" -p "${port}" >/dev/null 2>&1 && return 0
  fi

  # Método 2: nc netcat.
  if command -v nc >/dev/null 2>&1; then
    nc -z "${host}" "${port}" >/dev/null 2>&1 && return 0
  fi

  # Método 3: /dev/tcp en bash.
  if (echo > "/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
    return 0
  fi

  # Método 4: python socket check.
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
s = socket.socket()
s.settimeout(1.5)
try:
    s.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
    return $?
  fi

  return 1
}

build_sqlite_dsn() {
  local data_dir="${ROOT_DIR}/data"
  mkdir -p "${data_dir}"
  DATABASE_URL="sqlite:///${data_dir}/reco_trading.db"
  export DATABASE_URL
}

ensure_runtime_functions_available() {
  if ! declare -F build_sqlite_dsn >/dev/null 2>&1; then
    build_sqlite_dsn() {
      local data_dir="${ROOT_DIR}/data"
      mkdir -p "${data_dir}"
      DATABASE_URL="sqlite:///${data_dir}/reco_trading.db"
      export DATABASE_URL
    }
  fi
}

find_listening_pid_for_port() {
  local port="$1"

  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${port}" -sTCP:LISTEN -t 2>/dev/null | head -n 1
    return 0
  fi

  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "${port}" 2>/dev/null | awk '{print $1}'
    return 0
  fi

  if command -v ss >/dev/null 2>&1; then
    ss -tlnp 2>/dev/null | awk -v p=":${port}" '$4 ~ p {print $NF}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n 1
    return 0
  fi
}

release_dashboard_port_if_needed() {
  local port="$1"
  local pid=""

  pid="$(find_listening_pid_for_port "${port}" || true)"
  if [ -z "${pid}" ]; then
    return 0
  fi

  echo "⚠️  Puerto ${port} en uso por PID ${pid}. Intentando liberarlo..."
  if kill -TERM "${pid}" >/dev/null 2>&1; then
    # Esperar cierre limpio breve
    for _ in 1 2 3 4 5; do
      sleep 1
      if ! kill -0 "${pid}" >/dev/null 2>&1; then
        echo "✓ Puerto ${port} liberado (SIGTERM)."
        return 0
      fi
    done
  fi

  echo "⚠️  PID ${pid} no cerró a tiempo; forzando liberación (SIGKILL)..."
  if kill -KILL "${pid}" >/dev/null 2>&1; then
    sleep 1
  fi

  pid="$(find_listening_pid_for_port "${port}" || true)"
  if [ -n "${pid}" ]; then
    echo "Error: no se pudo liberar el puerto ${port} automáticamente."
    return 1
  fi

  echo "✓ Puerto ${port} liberado (SIGKILL)."
  return 0
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
ensure_runtime_functions_available

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

WEB_PORT="${WEB_DASHBOARD_PORT:-9000}"

# Asegurar puerto dashboard libre para mantener URL estable
release_dashboard_port_if_needed "${WEB_PORT}"

echo ""
echo "══════════════════════════════════════════════════"
echo "  ◈  RECO TRADING BOT  —  Iniciando..."
echo "  📊 Dashboard Web  :  http://127.0.0.1:${WEB_PORT}"
echo "  📡 Dashboard TUI  :  Esta consola"
echo "  ⌨️  Detener        :  Ctrl + C"
echo "══════════════════════════════════════════════════"
echo ""

exec python main.py
