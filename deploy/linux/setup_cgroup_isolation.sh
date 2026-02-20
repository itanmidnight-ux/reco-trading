#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${1:?Uso: $0 <servicio.systemd>}"
CPU_QUOTA="${2:-300%}"
MEMORY_MAX="${3:-8G}"
IO_WEIGHT="${4:-800}"
DROPIN_DIR="/etc/systemd/system/${SERVICE_NAME}.d"
DROPIN_FILE="${DROPIN_DIR}/30-cgroup-isolation.conf"

sudo mkdir -p "${DROPIN_DIR}"
cat <<CFG | sudo tee "${DROPIN_FILE}" >/dev/null
[Service]
CPUAccounting=true
MemoryAccounting=true
IOAccounting=true
CPUQuota=${CPU_QUOTA}
MemoryMax=${MEMORY_MAX}
IOWeight=${IO_WEIGHT}
TasksMax=8192
CFG

sudo systemctl daemon-reload
sudo systemctl restart "${SERVICE_NAME}"

echo "[OK] Límites cgroup aplicados en ${SERVICE_NAME}"
