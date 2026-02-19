#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3.11 >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3.11 python3.11-venv python3-pip
fi

sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib redis-server

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

sudo -u postgres psql -f scripts/init_db.sql || true
sudo systemctl enable --now postgresql
sudo systemctl enable --now redis-server

echo 'Instalación completada. Ejecuta ./run.sh'
