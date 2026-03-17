#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="$SCRIPT_DIR/runtime_config.json"
KEEP_RUNTIME_CONFIG="${KEEP_RUNTIME_CONFIG:-0}"
PREPARE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

log() {
  printf '[build_apk] %s\n' "$*"
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

load_env_file() {
  local env_file="$1"
  if [[ -f "$env_file" ]]; then
    # shellcheck disable=SC1090
    source "$env_file"
  fi
}

generate_runtime_config() {
  set -a
  load_env_file "$PROJECT_ROOT/.env"
  load_env_file "$SCRIPT_DIR/.env"
  set +a

  local public_url="${PUBLIC_API_URL:-}"
  local api_url="${RECO_API_URL:-}"
  local bootstrap_url="${RECO_BOOTSTRAP_URL:-}"
  local api_key="${RECO_API_KEY:-${API_AUTH_KEY:-change-me}}"

  if [[ -z "$public_url" ]]; then
    public_url="$(extract_ngrok_public_url || true)"
  fi

  if [[ -z "$api_url" && -n "$public_url" ]]; then
    api_url="$public_url"
  fi

  if [[ -z "$bootstrap_url" ]]; then
    bootstrap_url="${api_url:-http://10.0.2.2:8000}"
  fi

  if [[ -z "$api_url" ]]; then
    echo "❌ No se pudo inferir RECO_API_URL/PUBLIC_API_URL para la APK." >&2
    echo "   Configura .env con PUBLIC_API_URL o RECO_API_URL antes de compilar." >&2
    exit 1
  fi

  python3 - <<PY
import json
from pathlib import Path

config = {
    "RECO_AUTO_DISCOVERY": "true",
    "RECO_API_URL": "${api_url}".rstrip("/"),
    "RECO_BOOTSTRAP_URL": "${bootstrap_url}".rstrip("/"),
    "PUBLIC_API_URL": "${public_url}".rstrip("/"),
    "RECO_API_KEY": "${api_key}",
    "RECO_API_URL_CANDIDATES": [
        "${api_url}".rstrip("/"),
        "${public_url}".rstrip("/") if "${public_url}" else "",
        "http://10.0.2.2:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
}
config["RECO_API_URL_CANDIDATES"] = [c for c in config["RECO_API_URL_CANDIDATES"] if c]
Path("${CONFIG_FILE}").write_text(json.dumps(config, indent=2), encoding="utf-8")
print("runtime_config.json generado")
PY

  log "Configuración embebida para APK: api_url=${api_url} public_url=${public_url:-n/a}"
}

cleanup_runtime_config() {
  if [[ "$KEEP_RUNTIME_CONFIG" != "1" && -f "$CONFIG_FILE" ]]; then
    rm -f "$CONFIG_FILE"
    log "runtime_config.json removido del workspace local (ya fue usado para compilar)"
  fi
}

if [[ ! -x "$SCRIPT_DIR/build_android_auto.sh" ]]; then
  chmod +x "$SCRIPT_DIR/build_android_auto.sh"
fi

generate_runtime_config

if [[ "$PREPARE_ONLY" == "1" ]]; then
  log "Modo --prepare-only: configuración generada, se omite compilación."
  exit 0
fi

log "Build automático iniciado..."
"$SCRIPT_DIR/build_android_auto.sh" "$@"

APK_PATH="$(find "$SCRIPT_DIR/bin" -maxdepth 1 -type f -name '*.apk' | sort | tail -n 1 || true)"
if [[ -z "$APK_PATH" ]]; then
  echo "❌ No se encontró APK generado en app_android/bin"
  exit 1
fi

cleanup_runtime_config

echo "✅ APK listo y preconfigurado: $APK_PATH"
