#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

${SUDO} apt-get update
${SUDO} apt-get install -y python3.11 python3.11-venv python3-pip postgresql postgresql-contrib redis-server

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

${SUDO} systemctl enable --now postgresql
${SUDO} systemctl enable --now redis-server
${SUDO} -u postgres psql -f scripts/init_db.sql

echo 'Instalación completada. Edita .env con tus API keys y ejecuta ./run.sh'
