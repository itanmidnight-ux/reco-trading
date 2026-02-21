#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

DASHBOARD_CMD=(python -m streamlit run trading_system/dashboard/app.py --server.port 8080 --server.address 0.0.0.0)
KERNEL_CMD=(python main.py)

child_pids=()

cleanup() {
  local exit_code=${1:-0}

  for pid in "${child_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  for pid in "${child_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done

  exit "$exit_code"
}

trap 'cleanup 0' SIGINT SIGTERM
trap 'cleanup $?' EXIT

"${DASHBOARD_CMD[@]}" &
dashboard_pid=$!
child_pids+=("$dashboard_pid")

auto_kernel_loop() {
  while true; do
    "${KERNEL_CMD[@]}"
    sleep 1
  done
}

auto_kernel_loop &
kernel_pid=$!
child_pids+=("$kernel_pid")

wait "$dashboard_pid" "$kernel_pid"
