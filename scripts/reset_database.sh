#!/usr/bin/env bash
set -euo pipefail

DB_NAME="${DB_NAME:-reco_trading_prod}"
DB_USER="${DB_USER:-trading}"
DB_PASS="${DB_PASS:-trading_password}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
ADMIN_OS_USER="${ADMIN_OS_USER:-root}"
SCHEMA_PATH="${SCHEMA_PATH:-database/schema.sql}"

run_psql() {
  if sudo -u "$ADMIN_OS_USER" psql -h "$DB_HOST" -p "$DB_PORT" -Atqc 'SELECT 1' >/dev/null 2>&1; then
    sudo -u "$ADMIN_OS_USER" psql -h "$DB_HOST" -p "$DB_PORT" "$@"
  else
    # Fallback defensivo para distribuciones donde PostgreSQL usa el usuario de sistema postgres
    sudo -u postgres psql -h "$DB_HOST" -p "$DB_PORT" "$@"
  fi
}

echo '[1/6] Restarting PostgreSQL service...'
sudo service postgresql restart

echo '[2/6] Dropping existing database/user (if present)...'
run_psql -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS $DB_NAME;"
run_psql -v ON_ERROR_STOP=1 -c "DROP USER IF EXISTS $DB_USER;"

echo '[3/6] Creating role with deterministic password...'
run_psql -v ON_ERROR_STOP=1 -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';"

echo '[4/6] Creating database...'
run_psql -v ON_ERROR_STOP=1 -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"

echo '[5/6] Applying schema from' "$SCHEMA_PATH"
run_psql -v ON_ERROR_STOP=1 -d "$DB_NAME" -f "$SCHEMA_PATH"

echo '[6/6] Verifying connectivity using role' "$DB_USER"
PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -Atqc 'SELECT current_user, current_database();'

echo 'Database reset completed successfully.'
