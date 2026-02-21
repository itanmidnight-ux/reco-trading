#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

source config/database.env

POSTGRES_DSN="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

run_as_postgres() {
  if [[ -n "${SUDO}" ]]; then
    ${SUDO} -u postgres "$@"
  else
    runuser -u postgres -- "$@"
  fi
}

write_env_file() {
  local env_file="$1"
  local binance_api_key="$2"
  local binance_api_secret="$3"

  cat > "${env_file}" <<EOF
ENVIRONMENT=production
SYMBOL=BTC/USDT
TIMEFRAME=5m

# Binance Spot API (habilita whitelist IP desde tu cuenta Binance)
BINANCE_API_KEY=${binance_api_key}
BINANCE_API_SECRET=${binance_api_secret}
BINANCE_TESTNET=true

POSTGRES_DSN=${POSTGRES_DSN}
# Opcional: DSN admin para crear DB cuando no existe (ej: postgresql://postgres:postgres@localhost:5432/postgres)
POSTGRES_ADMIN_DSN=
REDIS_URL=redis://localhost:6379/0

RISK_PER_TRADE=0.01
MAX_DAILY_DRAWDOWN=0.03
MAX_CONSECUTIVE_LOSSES=3
ATR_STOP_MULTIPLIER=2.0
VOLATILITY_TARGET=0.20
CIRCUIT_BREAKER_VOLATILITY=0.08

MAKER_FEE=0.001
TAKER_FEE=0.001
SLIPPAGE_BPS=5
LOOP_INTERVAL_SECONDS=5
EOF
}

validate_allowed_env_vars() {
  local env_file="$1"
  local allowed_regex='^(ENVIRONMENT|SYMBOL|TIMEFRAME|BINANCE_API_KEY|BINANCE_API_SECRET|BINANCE_TESTNET|POSTGRES_DSN|POSTGRES_ADMIN_DSN|REDIS_URL|RISK_PER_TRADE|MAX_DAILY_DRAWDOWN|MAX_CONSECUTIVE_LOSSES|ATR_STOP_MULTIPLIER|VOLATILITY_TARGET|CIRCUIT_BREAKER_VOLATILITY|MAKER_FEE|TAKER_FEE|SLIPPAGE_BPS|LOOP_INTERVAL_SECONDS)$'
  local line_number=0

  while IFS= read -r line || [[ -n "${line}" ]]; do
    ((line_number += 1))

    if [[ -z "${line}" || "${line}" =~ ^# ]]; then
      continue
    fi

    if [[ ! "${line}" =~ ^[A-Z0-9_]+= ]]; then
      echo "ERROR: línea inválida en .env (${line_number}): ${line}" >&2
      exit 1
    fi

    local key="${line%%=*}"
    if [[ ! "${key}" =~ ${allowed_regex} ]]; then
      echo "ERROR: variable no permitida en .env: ${key}" >&2
      exit 1
    fi
  done < "${env_file}"
}

restart_postgres() {
  if command -v systemctl >/dev/null 2>&1; then
    ${SUDO} systemctl restart postgresql && return 0 || true
  fi

  if command -v service >/dev/null 2>&1; then
    ${SUDO} service postgresql restart && return 0 || true
  fi

  if command -v pg_ctlcluster >/dev/null 2>&1; then
    local cluster
    cluster=$(pg_lsclusters --no-header | awk 'NR==1 {print $1" "$2}')
    if [[ -n "${cluster}" ]]; then
      ${SUDO} pg_ctlcluster ${cluster} restart && return 0 || true
    fi
  fi

  echo "No se pudo reiniciar PostgreSQL automáticamente." >&2
  return 1
}

start_postgres() {
  if command -v systemctl >/dev/null 2>&1; then
    ${SUDO} systemctl enable postgresql || true
    ${SUDO} systemctl start postgresql && return 0 || true
  fi

  if command -v service >/dev/null 2>&1; then
    ${SUDO} service postgresql start && return 0 || true
  fi

  if command -v pg_ctlcluster >/dev/null 2>&1; then
    local cluster
    cluster=$(pg_lsclusters --no-header | awk 'NR==1 {print $1" "$2}')
    if [[ -n "${cluster}" ]]; then
      ${SUDO} pg_ctlcluster ${cluster} start && return 0 || true
    fi
  fi

  echo "No se pudo iniciar PostgreSQL automáticamente." >&2
  return 1
}

echo 'Instalando dependencias del sistema...'
${SUDO} apt-get update
${SUDO} apt-get install -y python3-venv postgresql postgresql-contrib redis-server

echo 'Configurando entorno virtual...'
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

if [[ ! -f .env ]]; then
  read -r -p "Enter Binance Testnet API Key: " BINANCE_API_KEY
  read -r -s -p "Enter Binance Testnet API Secret: " BINANCE_API_SECRET
  echo
  write_env_file ".env" "${BINANCE_API_KEY}" "${BINANCE_API_SECRET}"
fi

validate_allowed_env_vars ".env"

echo 'Asegurando servicios...'
start_postgres
if command -v systemctl >/dev/null 2>&1; then
  ${SUDO} systemctl enable redis-server || true
  ${SUDO} systemctl start redis-server || true
fi

echo 'Reconfigurando PostgreSQL en modo determinista y seguro...'

CLUSTER_INFO=$(pg_lsclusters --no-header | awk 'NR==1 {print $1" "$2}')
if [[ -z "${CLUSTER_INFO}" ]]; then
  echo "No se detectó cluster PostgreSQL activo." >&2
  exit 1
fi

PG_VERSION=$(echo "${CLUSTER_INFO}" | awk '{print $1}')
PG_CLUSTER=$(echo "${CLUSTER_INFO}" | awk '{print $2}')
PG_HBA_FILE="/etc/postgresql/${PG_VERSION}/${PG_CLUSTER}/pg_hba.conf"

if [[ ! -f "${PG_HBA_FILE}" ]]; then
  echo "No se encontró pg_hba.conf en ${PG_HBA_FILE}" >&2
  exit 1
fi

echo "Cluster detectado: PostgreSQL ${PG_VERSION} (${PG_CLUSTER})"

# Forzar autenticación correcta
${SUDO} sed -i -E "s#^local\s+all\s+postgres\s+.*#local   all   postgres   peer#" "${PG_HBA_FILE}"
${SUDO} sed -i -E "s#^local\s+all\s+all\s+.*#local   all   all   md5#" "${PG_HBA_FILE}"

if ! grep -q "^host\s\+all\s\+all\s\+127\.0\.0\.1/32" "${PG_HBA_FILE}"; then
  echo "host    all    all    127.0.0.1/32    md5" | ${SUDO} tee -a "${PG_HBA_FILE}" >/dev/null
fi

${SUDO} pg_ctlcluster ${PG_VERSION} ${PG_CLUSTER} restart

echo 'Validando peer auth...'
if ! ${SUDO} -u postgres psql -c "SELECT 1;" >/dev/null 2>&1; then
  echo "ERROR: peer authentication falló." >&2
  exit 1
fi

echo 'Recreando usuario y base completamente limpia...'

${SUDO} -u postgres psql <<SQL
DROP DATABASE IF EXISTS ${DB_NAME};
DROP ROLE IF EXISTS ${DB_USER};
CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASSWORD}';
CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};
SQL

${SUDO} -u postgres psql -d "${DB_NAME}" <<SQL
ALTER SCHEMA public OWNER TO ${DB_USER};
GRANT ALL ON SCHEMA public TO ${DB_USER};
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${DB_USER};
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ${DB_USER};
SQL

echo 'Validando autenticación TCP real...'

if ! PGPASSWORD="${DB_PASSWORD}" psql   -U "${DB_USER}"   -d "${DB_NAME}"   -h "${DB_HOST}"   -p "${DB_PORT}"   -c "SELECT 1;" >/dev/null 2>&1; then
  echo "ERROR: autenticación TCP falló." >&2
  exit 1
fi

echo 'PostgreSQL completamente sincronizado y validado.'

echo 'Instalación completada y validada correctamente.'
