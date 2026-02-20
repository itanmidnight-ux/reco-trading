#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${1:?Uso: $0 <servicio.systemd> <cpu_list>}"
CPU_LIST="${2:?Uso: $0 <servicio.systemd> <cpu_list>}"
DROPIN_DIR="/etc/systemd/system/${SERVICE_NAME}.d"
DROPIN_FILE="${DROPIN_DIR}/20-cpu-affinity.conf"

sudo mkdir -p "${DROPIN_DIR}"
cat <<CFG | sudo tee "${DROPIN_FILE}" >/dev/null
[Service]
CPUAffinity=${CPU_LIST}
CFG

sudo systemctl daemon-reload
sudo systemctl restart "${SERVICE_NAME}"

echo "[OK] CPUAffinity=${CPU_LIST} aplicado en ${SERVICE_NAME}"
