#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

source config/database.env

POSTGRES_DSN="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

ensure_env_file() {
  local env_file=".env"

  if [[ -f "${env_file}" ]]; then
    echo "Actualizando archivo ${env_file} con configuración base..."
  else
    echo "Creando archivo ${env_file} con configuración base..."
    cat > "${env_file}" <<ENV_TEMPLATE
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

# Dashboard
DASHBOARD_HOST=127.0.0.1
DASHBOARD_PORT=8080
AUTO_START_WEB=true
ENV_TEMPLATE
  fi

  upsert_env_var "${env_file}" "POSTGRES_DSN" "${POSTGRES_DSN}"
  upsert_env_var "${env_file}" "POSTGRES_ADMIN_DSN" "postgresql+asyncpg://postgres@${DB_HOST}:${DB_PORT}/postgres"
  upsert_env_var "${env_file}" "BINANCE_TESTNET" "true"
  upsert_env_var "${env_file}" "CONFIRM_MAINNET" "false"
  upsert_env_var "${env_file}" "ENVIRONMENT" "testnet"
  upsert_env_var "${env_file}" "RUNTIME_PROFILE" "paper"
  upsert_env_var "${env_file}" "REDIS_URL" "redis://localhost:6379/0"
}

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

run_as_postgres() {
  ${SUDO} -u postgres "$@"
}

echo 'Instalando dependencias del sistema...'
${SUDO} apt-get update
${SUDO} apt-get install -y python3-venv postgresql postgresql-contrib redis-server

echo 'Configurando entorno virtual...'
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip 

echo 'Asegurando servicios...'
${SUDO} systemctl enable postgresql || true
${SUDO} systemctl start postgresql || true
${SUDO} systemctl enable redis-server || true
${SUDO} systemctl start redis-server || true

echo 'Detectando cluster PostgreSQL activo...'

CLUSTER_INFO=$(pg_lsclusters --no-header | awk 'NR==1 {print $1" "$2}')
if [[ -z "${CLUSTER_INFO}" ]]; then
  echo "No se detectó cluster PostgreSQL activo." >&2
  exit 1
fi

PG_VERSION=$(echo "${CLUSTER_INFO}" | awk '{print $1}')
PG_CLUSTER=$(echo "${CLUSTER_INFO}" | awk '{print $2}')
PG_HBA_FILE="/etc/postgresql/${PG_VERSION}/${PG_CLUSTER}/pg_hba.conf"

echo "Cluster detectado: PostgreSQL ${PG_VERSION} (${PG_CLUSTER})"

if [[ ! -f "${PG_HBA_FILE}" ]]; then
  echo "No se encontró pg_hba.conf." >&2
  exit 1
fi

echo 'Configurando autenticación segura...'

${SUDO} sed -i -E "s#^local\s+all\s+postgres\s+.*#local   all   postgres   peer#" "${PG_HBA_FILE}"
${SUDO} sed -i -E "s#^local\s+all\s+all\s+.*#local   all   all   md5#" "${PG_HBA_FILE}"

if ! ${SUDO} grep -q "^host\s\+all\s\+all\s\+127\.0\.0\.1/32" "${PG_HBA_FILE}"; then
  echo "host    all    all    127.0.0.1/32    md5" | ${SUDO} tee -a "${PG_HBA_FILE}" >/dev/null
fi

${SUDO} pg_ctlcluster ${PG_VERSION} ${PG_CLUSTER} restart

echo 'Validando autenticación peer...'
if ! run_as_postgres psql -c "SELECT 1;" >/dev/null 2>&1; then
  echo "ERROR: peer authentication falló." >&2
  exit 1
fi

echo 'Eliminando todas las bases pertenecientes al usuario si existen...'

run_as_postgres psql -tAc "
SELECT datname FROM pg_database
WHERE pg_catalog.pg_get_userbyid(datdba) = '${DB_USER}'
" | while read -r db; do
  if [[ -n "$db" ]]; then
    echo "Eliminando base $db"
    run_as_postgres dropdb "$db"
  fi
done

echo 'Eliminando role si existe...'
run_as_postgres psql -c "DROP ROLE IF EXISTS ${DB_USER};"

echo 'Creando role limpio...'
run_as_postgres psql -c "CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASSWORD}';"

echo 'Creando base limpia...'
run_as_postgres createdb --owner="${DB_USER}" "${DB_NAME}"

echo 'Asignando permisos y ownership...'
run_as_postgres psql -d "${DB_NAME}" -c "ALTER SCHEMA public OWNER TO ${DB_USER};"
run_as_postgres psql -d "${DB_NAME}" -c "GRANT ALL ON SCHEMA public TO ${DB_USER};"
run_as_postgres psql -d "${DB_NAME}" -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${DB_USER};"
run_as_postgres psql -d "${DB_NAME}" -c "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ${DB_USER};"

echo 'Validando autenticación TCP real...'

if ! PGPASSWORD="${DB_PASSWORD}" psql \
  -U "${DB_USER}" \
  -d "${DB_NAME}" \
  -h "${DB_HOST}" \
  -p "${DB_PORT}" \
  -c "SELECT 1;" >/dev/null 2>&1; then
  echo "ERROR: autenticación TCP falló." >&2
  exit 1
fi

ensure_env_file

echo 'PostgreSQL sincronizado y validado correctamente.'
echo "Archivo .env sincronizado (actualiza manualmente BINANCE_API_KEY y BINANCE_API_SECRET)."
echo 'Instalación completada con éxito.'
