#!/usr/bin/env bash
set -euo pipefail

TARGET_USER="${1:-trading}"
LIMITS_FILE="/etc/security/limits.d/reco-trading.conf"

cat <<CFG | sudo tee "${LIMITS_FILE}" >/dev/null
${TARGET_USER} soft nofile 262144
${TARGET_USER} hard nofile 262144
${TARGET_USER} soft nproc  65535
${TARGET_USER} hard nproc  65535
CFG

echo "[OK] Límites persistentes escritos en ${LIMITS_FILE}"
echo "[INFO] Abre una nueva sesión para validar con: ulimit -n && ulimit -u"
