#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -x "$SCRIPT_DIR/build_android_auto.sh" ]]; then
  chmod +x "$SCRIPT_DIR/build_android_auto.sh"
fi

echo "[Reco Trading Android] Build automático iniciado..."
"$SCRIPT_DIR/build_android_auto.sh" "$@"

APK_PATH="$(find "$SCRIPT_DIR/bin" -maxdepth 1 -type f -name '*.apk' | sort | tail -n 1 || true)"
if [[ -z "$APK_PATH" ]]; then
  echo "❌ No se encontró APK generado en app_android/bin"
  exit 1
fi

echo "✅ APK listo: $APK_PATH"
