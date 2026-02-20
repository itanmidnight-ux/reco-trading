#!/usr/bin/env bash
set -euo pipefail

SYSCTL_FILE="/etc/sysctl.d/99-reco-trading-net.conf"

cat <<CFG | sudo tee "${SYSCTL_FILE}" >/dev/null
# Reco Trading baseline (Linux kernel 6.x)
net.core.somaxconn = 4096
net.core.netdev_max_backlog = 16384
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_tw_reuse = 1
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_rmem = 4096 262144 33554432
net.ipv4.tcp_wmem = 4096 262144 33554432
net.core.rmem_max = 33554432
net.core.wmem_max = 33554432
net.ipv4.tcp_keepalive_time = 120
net.ipv4.tcp_keepalive_intvl = 30
net.ipv4.tcp_keepalive_probes = 5
vm.max_map_count = 1048576
CFG

sudo sysctl --system >/dev/null
echo "[OK] Parámetros de red aplicados desde ${SYSCTL_FILE}"
