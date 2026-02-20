#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
if [[ -z "${ROLE}" ]]; then
  echo "Uso: $0 <orchestrator|trading-worker|evolution-worker|architecture-search-worker|self-healing-worker>"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_CMD=""
case "${ROLE}" in
  orchestrator)
    DEFAULT_CMD="python scripts/live_trading.py"
    ;;
  trading-worker)
    DEFAULT_CMD="python scripts/live_trading.py"
    ;;
  evolution-worker)
    DEFAULT_CMD="python scripts/live_trading.py"
    ;;
  architecture-search-worker)
    DEFAULT_CMD="python -c 'import time; print(\"architecture-search-worker placeholder\"); time.sleep(3600)'"
    ;;
  self-healing-worker)
    DEFAULT_CMD="python -c 'import time; print(\"self-healing-worker placeholder\"); time.sleep(3600)'"
    ;;
  *)
    echo "Role no soportado: ${ROLE}"
    exit 2
    ;;
esac

ENV_NAME="$(echo "${ROLE}" | tr '[:lower:]-' '[:upper:]_')_CMD"
ROLE_CMD="${!ENV_NAME:-${DEFAULT_CMD}}"

exec /bin/bash -lc "${ROLE_CMD}"
