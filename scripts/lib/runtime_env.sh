#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

upsert_env_var() {
  local env_file="$1"
  local key="$2"
  local value="$3"

  if [[ ! -f "${env_file}" ]]; then
    touch "${env_file}"
  fi

  if grep -qE "^${key}=" "${env_file}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${env_file}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${env_file}"
  fi
}

load_dotenv_file() {
  local env_file="$1"
  if [[ -f "${env_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
}

load_runtime_env() {
  cd "${ROOT_DIR}"
  load_dotenv_file .env
  load_dotenv_file config/database.env
}

build_postgres_dsn_from_config() {
  if [[ -n "${DB_USER:-}" && -n "${DB_PASSWORD:-}" && -n "${DB_HOST:-}" && -n "${DB_PORT:-}" && -n "${DB_NAME:-}" ]]; then
    export POSTGRES_DSN="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
    export POSTGRES_ADMIN_DSN="postgresql+asyncpg://postgres@${DB_HOST}:${DB_PORT}/postgres"
    upsert_env_var .env POSTGRES_DSN "${POSTGRES_DSN}"
    upsert_env_var .env POSTGRES_ADMIN_DSN "${POSTGRES_ADMIN_DSN}"
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

postgres_application_reachable() {
  python - <<'PY'
from __future__ import annotations

import asyncio
import os
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def main() -> int:
    dsn = os.environ.get("POSTGRES_DSN", "").strip()
    if not dsn:
        return 2

    engine = create_async_engine(dsn, echo=False, future=True)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        return 1
    finally:
        await engine.dispose()
    return 0


raise SystemExit(asyncio.run(main()))
PY
}
