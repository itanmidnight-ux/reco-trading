#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

${SUDO} apt-get update

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

${SUDO} systemctl enable --now postgresql
${SUDO} systemctl enable --now redis-server
${SUDO} -u postgres psql -f scripts/init_db.sql

echo 'Instalación completada. Edita .env con tus API keys y ejecuta ./run.sh'
