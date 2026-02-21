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

echo 'PostgreSQL sincronizado y validado correctamente.'
echo 'Instalación completada con éxito.'
