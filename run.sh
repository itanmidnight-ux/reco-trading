#!/usr/bin/env bash
set -euo pipefail

BACKEND_PID=""
NGROK_PID=""
MONITOR_PID=""

: "${NGROK_CHECK_INTERVAL_SECONDS:=8}"
: "${NGROK_MAX_RESTARTS_PER_HOUR:=30}"
: "${NGROK_REGION:=}"
: "${NGROK_DOMAIN:=}"

cleanup() {
  for pid in "${MONITOR_PID}" "${NGROK_PID}" "${BACKEND_PID}"; do
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT INT TERM

upsert_env_var() {
  local env_file="$1"
  local key="$2"
  local value="$3"

  if [ ! -f "${env_file}" ]; then
    touch "${env_file}"
  fi

  if grep -qE "^${key}=" "${env_file}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${env_file}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${env_file}"
  fi
}

sync_mode_to_env() {
  local mode_option="$1"
  local env_file=".env"

  if [ ! -f "${env_file}" ]; then
    touch "${env_file}"
  fi

  if [ "${mode_option}" = "1" ]; then
    upsert_env_var "${env_file}" "BINANCE_TESTNET" "true"
    upsert_env_var "${env_file}" "CONFIRM_MAINNET" "false"
    upsert_env_var "${env_file}" "ENVIRONMENT" "testnet"
    upsert_env_var "${env_file}" "RUNTIME_PROFILE" "paper"
  elif [ "${mode_option}" = "2" ]; then
    upsert_env_var "${env_file}" "BINANCE_TESTNET" "false"
    upsert_env_var "${env_file}" "CONFIRM_MAINNET" "true"
    upsert_env_var "${env_file}" "ENVIRONMENT" "production"
    upsert_env_var "${env_file}" "RUNTIME_PROFILE" "production"
  fi
}

log_info() {
  echo "[run.sh] $*"
}

extract_ngrok_public_url() {
  python3 - <<'PY'
import json
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception:
    print("")
    raise SystemExit(0)

for tunnel in payload.get("tunnels", []):
    public_url = str(tunnel.get("public_url", "")).strip()
    if public_url.startswith("https://"):
        print(public_url)
        raise SystemExit(0)

print("")
PY
}

wait_for_api_ready() {
  local max_attempts="$1"
  local attempt=1

  while [ "${attempt}" -le "${max_attempts}" ]; do
    if curl -fsS --max-time 2 "http://127.0.0.1:8000/openapi.json" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    attempt=$((attempt + 1))
  done
  return 1
}

wait_for_ngrok_url() {
  local max_attempts="$1"
  local attempt=1

  while [ "${attempt}" -le "${max_attempts}" ]; do
    local extracted
    extracted="$(extract_ngrok_public_url || true)"
    if [[ "${extracted}" =~ ^https:// ]]; then
      printf '%s\n' "${extracted}"
      return 0
    fi
    sleep 1
    attempt=$((attempt + 1))
  done

  return 1
}

start_ngrok() {
  local args=(http 8000 --log=stdout)
  if [ -n "${NGROK_REGION}" ]; then
    args+=(--region "${NGROK_REGION}")
  fi
  if [ -n "${NGROK_DOMAIN}" ]; then
    args+=(--domain "${NGROK_DOMAIN}")
  fi
  if [ -n "${NGROK_AUTHTOKEN:-}" ]; then
    args+=(--authtoken "${NGROK_AUTHTOKEN}")
  fi

  ngrok "${args[@]}" > ngrok.log 2>&1 &
  NGROK_PID=$!
  log_info "ngrok iniciado (pid=${NGROK_PID})"
}

restart_ngrok_if_unhealthy() {
  local current_url="$1"
  local restarted=0

  if ! kill -0 "${NGROK_PID}" 2>/dev/null; then
    restarted=1
  elif ! curl -fsS --max-time 5 "${current_url}/public-url" >/dev/null 2>&1; then
    restarted=1
  fi

  if [ "${restarted}" -eq 1 ]; then
    log_info "ngrok no saludable; reiniciando túnel"
    if [ -n "${NGROK_PID}" ] && kill -0 "${NGROK_PID}" 2>/dev/null; then
      kill "${NGROK_PID}" 2>/dev/null || true
    fi
    start_ngrok
    local new_url
    new_url="$(wait_for_ngrok_url 45 || true)"
    if [[ "${new_url}" =~ ^https:// ]]; then
      upsert_env_var .env PUBLIC_API_URL "${new_url}"
      export PUBLIC_API_URL="${new_url}"
      log_info "PUBLIC_API_URL actualizada tras reinicio: ${new_url}"
      printf '%s\n' "${new_url}"
      return 0
    fi
  fi

  printf '%s\n' "${current_url}"
  return 0
}

monitor_tunnel_and_backend() {
  local current_url="$1"
  local restart_counter=0
  local window_start
  window_start="$(date +%s)"

  while kill -0 "${BACKEND_PID}" 2>/dev/null; do
    sleep "${NGROK_CHECK_INTERVAL_SECONDS}"

    local now
    now="$(date +%s)"
    if [ $((now - window_start)) -ge 3600 ]; then
      restart_counter=0
      window_start="${now}"
    fi

    local updated_url
    updated_url="$(restart_ngrok_if_unhealthy "${current_url}")"
    if [ "${updated_url}" != "${current_url}" ]; then
      current_url="${updated_url}"
      restart_counter=$((restart_counter + 1))
    fi

    local latest_url
    latest_url="$(extract_ngrok_public_url || true)"
    if [[ "${latest_url}" =~ ^https:// ]] && [ "${latest_url}" != "${current_url}" ]; then
      current_url="${latest_url}"
      upsert_env_var .env PUBLIC_API_URL "${current_url}"
      export PUBLIC_API_URL="${current_url}"
      log_info "PUBLIC_API_URL actualizada tras cambio de túnel: ${current_url}"
    fi

    if [ "${restart_counter}" -gt "${NGROK_MAX_RESTARTS_PER_HOUR}" ]; then
      log_info "Demasiados reinicios de ngrok en 1h (${restart_counter}). Saliendo por seguridad."
      break
    fi
  done
}

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "Aviso: .venv/bin/activate no existe. Usando Python del sistema."
fi

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -z "${POSTGRES_DSN:-}" ] && [ -f config/database.env ]; then
  set -a
  # shellcheck disable=SC1091
  source config/database.env
  set +a
  if [ -n "${DB_USER:-}" ] && [ -n "${DB_PASSWORD:-}" ] && [ -n "${DB_HOST:-}" ] && [ -n "${DB_PORT:-}" ] && [ -n "${DB_NAME:-}" ]; then
    export POSTGRES_DSN="postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
    upsert_env_var .env POSTGRES_DSN "${POSTGRES_DSN}"
  fi
fi

MODE_OPTION="${RUN_MODE:-}"
if [ -z "${MODE_OPTION}" ]; then
  echo "Seleccione modo de ejecución:"
  echo "1) Binance Testnet (Sandbox - ÓRDENES REALES EN TESTNET)"
  echo "2) Binance Producción Real (Mainnet - Dinero real)"
  read -r -p "Ingrese opción (1 o 2): " MODE_OPTION
fi

if [ "$MODE_OPTION" = "1" ]; then
  export BINANCE_TESTNET=true
  export CONFIRM_MAINNET=false
  export ENVIRONMENT=testnet
  export RUNTIME_PROFILE=paper
  sync_mode_to_env "$MODE_OPTION"
  echo "Modo TESTNET activado."
elif [ "$MODE_OPTION" = "2" ]; then
  export BINANCE_TESTNET=false
  if [ "${RUN_MODE:-}" = "2" ]; then
    CONFIRM="CONFIRMAR"
  else
    read -r -p "⚠️  Está a punto de operar con dinero real. Escriba CONFIRMAR para continuar: " CONFIRM
  fi
  if [ "$CONFIRM" != "CONFIRMAR" ]; then
    echo "Operación cancelada."
    exit 1
  fi
  if [ -z "${BINANCE_API_KEY:-}" ] || [ -z "${BINANCE_API_SECRET:-}" ]; then
    echo "Error: BINANCE_API_KEY y BINANCE_API_SECRET son obligatorios para producción real."
    exit 1
  fi
  export CONFIRM_MAINNET=true
  export ENVIRONMENT=production
  export RUNTIME_PROFILE=production
  sync_mode_to_env "$MODE_OPTION"
  echo "Modo PRODUCCIÓN REAL activado."
else
  echo "Opción inválida."
  exit 1
fi

missing_vars=()
for required_var in BINANCE_API_KEY BINANCE_API_SECRET POSTGRES_DSN; do
  if [ -z "${!required_var:-}" ]; then
    missing_vars+=("${required_var}")
  fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
  echo "Error: faltan variables obligatorias: ${missing_vars[*]}"
  exit 1
fi

if ! command -v ngrok >/dev/null 2>&1; then
  echo "Error: ngrok no está instalado o no está en PATH. Instálelo desde https://ngrok.com/download"
  exit 1
fi

log_info "Iniciando backend (bot + API)..."
python3 main.py > backend.log 2>&1 &
BACKEND_PID=$!

if ! wait_for_api_ready 60; then
  echo "Error: la API no respondió en http://127.0.0.1:8000/openapi.json"
  echo "Revise backend.log para más detalles."
  exit 1
fi
log_info "API disponible en puerto 8000"
log_info "Dashboard web seguro disponible localmente en login: http://127.0.0.1:8000/"

start_ngrok

PUBLIC_URL="$(wait_for_ngrok_url 60 || true)"
if [[ ! "${PUBLIC_URL}" =~ ^https:// ]]; then
  echo "Error: no se pudo obtener una URL pública válida de ngrok. Revise ngrok.log"
  exit 1
fi

upsert_env_var .env PUBLIC_API_URL "${PUBLIC_URL}"
export PUBLIC_API_URL="${PUBLIC_URL}"
log_info "URL pública detectada: ${PUBLIC_API_URL}"
log_info "Dashboard web seguro público (login): ${PUBLIC_API_URL}/"
if [ -z "${NGROK_DOMAIN}" ]; then
  log_info "Aviso: para mantener un enlace fijo, configure NGROK_DOMAIN con un dominio reservado de ngrok."
fi

monitor_tunnel_and_backend "${PUBLIC_URL}" &
MONITOR_PID=$!

wait "${BACKEND_PID}"
kill "${MONITOR_PID}" 2>/dev/null || true
