#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/lib/runtime_env.sh"

if [[ ! -f config/database.env ]]; then
  cat > config/database.env <<'ENV_TEMPLATE'
# Optional helper values to compose POSTGRES_DSN in run.sh
# Fill locally if you want run.sh to auto-build POSTGRES_DSN.
DB_USER=trading
DB_PASSWORD=trading123
DB_NAME=reco_trading_prod
DB_HOST=localhost
DB_PORT=5432
ENV_TEMPLATE
fi

load_dotenv_file config/database.env

required_db_vars=(DB_USER DB_PASSWORD DB_NAME DB_HOST DB_PORT)
missing_db_vars=()
for required_var in "${required_db_vars[@]}"; do
  if [[ -z "${!required_var:-}" ]]; then
    missing_db_vars+=("${required_var}")
  fi
done

if [[ ${#missing_db_vars[@]} -gt 0 ]]; then
  echo "Error: faltan variables en config/database.env: ${missing_db_vars[*]}" >&2
  exit 1
fi

case "${DB_HOST}" in
  localhost|127.0.0.1|::1) ;;
  *)
    echo "Error: scripts/postgres/bootstrap_local_postgres.sh solo puede auto-configurar PostgreSQL local (DB_HOST actual: ${DB_HOST})." >&2
    exit 1
    ;;
esac

if [[ "${EUID}" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

run_as_postgres() {
  local quoted_cmd
  if [[ -n "${SUDO}" ]]; then
    ${SUDO} -u postgres "$@"
    return
  fi

  if [[ "${EUID}" -eq 0 ]]; then
    printf -v quoted_cmd '%q ' "$@"
    su - postgres -c "${quoted_cmd% }"
    return
  fi

  echo "Error: se requiere sudo o root para configurar PostgreSQL automáticamente." >&2
  exit 1
}

install_postgres_if_missing() {
  if command -v psql >/dev/null 2>&1 && command -v pg_lsclusters >/dev/null 2>&1; then
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Error: PostgreSQL no está instalado y este entorno no soporta instalación automática con apt-get." >&2
    exit 1
  fi

  echo "Instalando PostgreSQL automáticamente..."
  export DEBIAN_FRONTEND=noninteractive
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y postgresql postgresql-contrib
}

ensure_postgres_service() {
  echo "Asegurando servicio PostgreSQL..."

  if command -v systemctl >/dev/null 2>&1; then
    ${SUDO} systemctl enable postgresql >/dev/null 2>&1 || true
    ${SUDO} systemctl start postgresql >/dev/null 2>&1 || true
  elif command -v service >/dev/null 2>&1; then
    ${SUDO} service postgresql start >/dev/null 2>&1 || true
  fi

  if ! command -v pg_lsclusters >/dev/null 2>&1; then
    echo "Error: pg_lsclusters no está disponible tras la instalación." >&2
    exit 1
  fi
}

detect_cluster() {
  local cluster_info
  cluster_info="$(pg_lsclusters --no-header | awk 'NR==1 {print $1" " $2}')"
  if [[ -z "${cluster_info}" ]]; then
    echo "Error: no se detectó ningún cluster PostgreSQL." >&2
    exit 1
  fi

  PG_VERSION="$(echo "${cluster_info}" | awk '{print $1}')"
  PG_CLUSTER="$(echo "${cluster_info}" | awk '{print $2}')"
  PG_HBA_FILE="/etc/postgresql/${PG_VERSION}/${PG_CLUSTER}/pg_hba.conf"
}

configure_local_auth() {
  if [[ ! -f "${PG_HBA_FILE}" ]]; then
    echo "Error: no se encontró ${PG_HBA_FILE}" >&2
    exit 1
  fi

  echo "Configurando autenticación local de PostgreSQL..."
  ${SUDO} sed -i -E "s#^local\s+all\s+postgres\s+.*#local   all   postgres   peer#" "${PG_HBA_FILE}"
  ${SUDO} sed -i -E "s#^local\s+all\s+all\s+.*#local   all   all   md5#" "${PG_HBA_FILE}"

  if ! ${SUDO} grep -q "^host\s\+all\s\+all\s\+127\.0\.0\.1/32" "${PG_HBA_FILE}"; then
    echo "host    all    all    127.0.0.1/32    md5" | ${SUDO} tee -a "${PG_HBA_FILE}" >/dev/null
  fi

  if ! ${SUDO} grep -q "^host\s\+all\s\+all\s\+::1/128" "${PG_HBA_FILE}"; then
    echo "host    all    all    ::1/128    md5" | ${SUDO} tee -a "${PG_HBA_FILE}" >/dev/null
  fi

  ${SUDO} pg_ctlcluster "${PG_VERSION}" "${PG_CLUSTER}" restart >/dev/null 2>&1 || true
}

ensure_role_and_database() {
  echo "Asegurando role y base de datos..."

  if ! run_as_postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1; then
    run_as_postgres psql -c "CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASSWORD}';" >/dev/null
  else
    run_as_postgres psql -c "ALTER ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASSWORD}';" >/dev/null
  fi

  if ! run_as_postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
    run_as_postgres createdb --owner="${DB_USER}" "${DB_NAME}"
  fi

  run_as_postgres psql -d "${DB_NAME}" -c "ALTER SCHEMA public OWNER TO ${DB_USER};" >/dev/null
  run_as_postgres psql -d "${DB_NAME}" -c "GRANT ALL ON SCHEMA public TO ${DB_USER};" >/dev/null
  run_as_postgres psql -d "${DB_NAME}" -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${DB_USER};" >/dev/null
  run_as_postgres psql -d "${DB_NAME}" -c "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ${DB_USER};" >/dev/null
}

validate_tcp_auth() {
  echo "Validando conexión TCP a PostgreSQL..."
  if ! PGPASSWORD="${DB_PASSWORD}" psql \
    -U "${DB_USER}" \
    -d "${DB_NAME}" \
    -h "${DB_HOST}" \
    -p "${DB_PORT}" \
    -c "SELECT 1;" >/dev/null 2>&1; then
    echo "Error: la validación TCP de PostgreSQL falló." >&2
    exit 1
  fi
}

sync_env() {
  build_postgres_dsn_from_config
}

install_postgres_if_missing
ensure_postgres_service
detect_cluster
configure_local_auth
ensure_role_and_database
validate_tcp_auth
sync_env

echo "PostgreSQL listo y .env sincronizado."
