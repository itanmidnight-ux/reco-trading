#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Falta comando requerido: $1"; exit 1; }
}

require_cmd python
require_cmd docker

python - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit('Se requiere Python 3.11+')
print('Python OK')
PY

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

docker compose up -d postgres redis
sleep 6

python -m compileall trading_system tests scripts
pytest -q

echo "Instalación completa. Inicia el sistema con: source .venv/bin/activate && python trading_system/app/main.py"
