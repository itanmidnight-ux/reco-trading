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

upsert_env_key() {
  local key="$1"
  local value="$2"
  local env_file="$3"

  if [[ ! -f "${env_file}" ]]; then
    touch "${env_file}"
  fi

  if rg -q "^${key}=" "${env_file}"; then
    sed -i "s#^${key}=.*#${key}=${value}#" "${env_file}"
  else
    printf '\n%s=%s\n' "${key}" "${value}" >> "${env_file}"
  fi
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
pip install -r requirements.txt --no-cache-dir

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

upsert_env_key "POSTGRES_DSN" "${POSTGRES_DSN}" ".env"

echo 'Asegurando servicios...'
start_postgres
if command -v systemctl >/dev/null 2>&1; then
  ${SUDO} systemctl enable redis-server || true
  ${SUDO} systemctl start redis-server || true
fi

echo 'Sincronizando usuario y contraseña de PostgreSQL...'
run_as_postgres psql <<SQL
DO
\$do\$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles WHERE rolname = '${DB_USER}'
   ) THEN
      CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASSWORD}';
   ELSE
      ALTER ROLE ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';
   END IF;
END
\$do\$;
SQL

echo 'Creando base de datos si no existe...'
if ! run_as_postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | rg -q '^1$'; then
  run_as_postgres createdb --owner="${DB_USER}" "${DB_NAME}"
fi

echo 'Aplicando ownership y permisos de schema public...'
run_as_postgres psql -d "${DB_NAME}" <<SQL
GRANT ALL ON SCHEMA public TO ${DB_USER};
ALTER SCHEMA public OWNER TO ${DB_USER};
SQL

echo 'Ajustando pg_hba.conf...'
PG_HBA_FILE=$(find /etc/postgresql -path '*/main/pg_hba.conf' | head -n 1)
if [[ -z "${PG_HBA_FILE}" ]]; then
  echo 'No se encontró pg_hba.conf en /etc/postgresql/*/main/pg_hba.conf' >&2
  exit 1
fi

${SUDO} sed -i -E "s#^local\s+all\s+all\s+.*#local   all   all   md5#" "${PG_HBA_FILE}"
if rg -q "^local\s+all\s+all\s+md5$" "${PG_HBA_FILE}"; then
  true
else
  echo "local   all   all   md5" | ${SUDO} tee -a "${PG_HBA_FILE}" >/dev/null
fi

${SUDO} sed -i -E "s#^host\s+all\s+all\s+127\.0\.0\.1/32\s+.*#host    all   all   127.0.0.1/32   md5#" "${PG_HBA_FILE}"
if rg -q "^host\s+all\s+all\s+127\.0\.0\.1/32\s+md5$" "${PG_HBA_FILE}"; then
  true
else
  echo "host    all   all   127.0.0.1/32   md5" | ${SUDO} tee -a "${PG_HBA_FILE}" >/dev/null
fi

restart_postgres

echo 'Validando conexión con credenciales sincronizadas...'
if ! PGPASSWORD="${DB_PASSWORD}" psql -U "${DB_USER}" -d "${DB_NAME}" -h "${DB_HOST}" -p "${DB_PORT}" -c 'SELECT 1;' ; then
  echo "ERROR: no fue posible autenticar con ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}" >&2
  exit 1
fi

echo 'Instalación completada y validada correctamente.'
