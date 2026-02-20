#!/usr/bin/env bash
set -euo pipefail

PY_MIN="3.11"
REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
POSTGRES_HOST="${POSTGRES_HOST:-127.0.0.1}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

python3 - <<'PY'
import platform
import sys
min_version=(3,11)
if sys.version_info < min_version:
    raise SystemExit(f"[FAIL] Python {platform.python_version()} < {min_version[0]}.{min_version[1]}")
print(f"[OK] Python {platform.python_version()} compatible")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader | sed 's/^/[OK] CUDA /'
else
  echo "[WARN] nvidia-smi no disponible (CUDA opcional pero recomendado para inferencia GPU)"
fi

if command -v redis-cli >/dev/null 2>&1; then
  redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping | sed 's/^/[OK] Redis /'
else
  echo "[WARN] redis-cli no instalado; no se pudo validar Redis"
fi

if command -v pg_isready >/dev/null 2>&1; then
  pg_isready -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" | sed 's/^/[OK] PostgreSQL /'
else
  echo "[WARN] pg_isready no instalado; no se pudo validar PostgreSQL"
fi
