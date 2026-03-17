#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv_android}"
STATE_FILE="${STATE_FILE:-$SCRIPT_DIR/.android_build_state}"
FORCE_CLEAN=0
SKIP_INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-clean)
      FORCE_CLEAN=1
      ;;
    --no-install)
      SKIP_INSTALL=1
      ;;
    -h|--help)
      cat <<'USAGE'
Uso: ./build_android_auto.sh [opciones]

Opciones:
  --force-clean   Fuerza recompilación completa (buildozer android clean)
  --no-install    Omite instalación/verificación de dependencias
  -h, --help      Muestra esta ayuda
USAGE
      exit 0
      ;;
    *)
      echo "❌ Opción no reconocida: $1"
      exit 1
      ;;
  esac
  shift
done

log() {
  printf '\n[%s] %s\n' "$(date +'%H:%M:%S')" "$*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

run_as_root() {
  if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    "$@"
  elif have_cmd sudo; then
    sudo "$@"
  else
    echo "❌ Se requieren permisos de root para instalar paquetes y no se encontró sudo."
    exit 1
  fi
}

detect_pkg_manager() {
  for pm in apt-get dnf yum pacman zypper apk; do
    if have_cmd "$pm"; then
      echo "$pm"
      return 0
    fi
  done
  return 1
}

install_packages() {
  local pm="$1"
  shift
  local pkgs=("$@")

  case "$pm" in
    apt-get)
      run_as_root apt-get update -y
      run_as_root apt-get install -y "${pkgs[@]}"
      ;;
    dnf)
      run_as_root dnf install -y "${pkgs[@]}"
      ;;
    yum)
      run_as_root yum install -y "${pkgs[@]}"
      ;;
    pacman)
      run_as_root pacman -Sy --noconfirm --needed "${pkgs[@]}"
      ;;
    zypper)
      run_as_root zypper --non-interactive install --no-recommends "${pkgs[@]}"
      ;;
    apk)
      run_as_root apk add --no-cache "${pkgs[@]}"
      ;;
    *)
      echo "❌ Gestor de paquetes no soportado: $pm"
      exit 1
      ;;
  esac
}

install_system_deps_if_needed() {
  local pm
  pm="$(detect_pkg_manager)" || {
    echo "❌ No pude detectar gestor de paquetes. Instala manualmente Python 3, JDK 17, git y herramientas de compilación."
    exit 1
  }

  log "Gestor de paquetes detectado: $pm"

  local cmd_deps=(git zip unzip java python3 pip3)
  local missing_cmds=()
  for dep in "${cmd_deps[@]}"; do
    if ! have_cmd "$dep"; then
      missing_cmds+=("$dep")
    fi
  done

  if [[ ${#missing_cmds[@]} -eq 0 ]]; then
    log "Dependencias base ya instaladas, no se reinstalan."
    return 0
  fi

  log "Faltan comandos: ${missing_cmds[*]}"

  case "$pm" in
    apt-get)
      install_packages "$pm" \
        build-essential git zip unzip openjdk-17-jdk \
        python3 python3-venv python3-pip python3-dev \
        libffi-dev libssl-dev libsqlite3-dev zlib1g-dev
      ;;
    dnf|yum)
      install_packages "$pm" \
        gcc gcc-c++ make git zip unzip java-17-openjdk-devel \
        python3 python3-pip python3-devel \
        libffi-devel openssl-devel sqlite-devel zlib-devel
      ;;
    pacman)
      install_packages "$pm" \
        base-devel git zip unzip jdk17-openjdk \
        python python-pip
      ;;
    zypper)
      install_packages "$pm" \
        -t pattern devel_basis
      install_packages "$pm" \
        git zip unzip java-17-openjdk-devel \
        python3 python3-pip python3-devel libffi-devel libopenssl-devel
      ;;
    apk)
      install_packages "$pm" \
        build-base git zip unzip openjdk17 \
        python3 py3-pip python3-dev libffi-dev openssl-dev
      ;;
  esac
}

create_or_update_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creando entorno virtual en $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  else
    log "Entorno virtual detectado en $VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  log "Actualizando pip/setuptools/wheel"
  python -m pip install --upgrade pip setuptools wheel

  local pip_packages=(cython buildozer kivy requests)
  log "Instalando/actualizando herramientas Python: ${pip_packages[*]}"
  python -m pip install --upgrade "${pip_packages[@]}"
}

hash_build_inputs() {
  local files=(buildozer.spec requirements.txt)
  local hasher
  hasher="$(command -v sha256sum || command -v shasum)"
  if [[ -z "$hasher" ]]; then
    echo "no-hash-tool"
    return
  fi

  local existing=()
  for f in "${files[@]}"; do
    [[ -f "$f" ]] && existing+=("$f")
  done

  if [[ ${#existing[@]} -eq 0 ]]; then
    echo "no-input-files"
    return
  fi

  if [[ "$hasher" == *"sha256sum" ]]; then
    sha256sum "${existing[@]}" | sha256sum | awk '{print $1}'
  else
    shasum -a 256 "${existing[@]}" | shasum -a 256 | awk '{print $1}'
  fi
}

needs_clean_build() {
  [[ $FORCE_CLEAN -eq 1 ]] && return 0
  [[ ! -d .buildozer ]] && return 0

  local current_hash old_hash
  current_hash="$(hash_build_inputs)"
  old_hash=""
  [[ -f "$STATE_FILE" ]] && old_hash="$(cat "$STATE_FILE")"

  [[ "$current_hash" != "$old_hash" ]]
}

run_build() {
  if [[ ! -f buildozer.spec ]]; then
    echo "❌ No se encontró buildozer.spec en $SCRIPT_DIR"
    exit 1
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  if needs_clean_build; then
    log "Se detectaron cambios de configuración o no existe build previo. Recompilación completa..."
    buildozer android clean
  else
    log "Sin cambios estructurales. Build incremental..."
  fi

  log "Compilando APK (debug)..."
  buildozer -v android debug

  hash_build_inputs > "$STATE_FILE"

  local apk
  apk="$(find bin -maxdepth 1 -type f -name '*.apk' | sort | tail -n 1 || true)"
  if [[ -n "$apk" ]]; then
    log "✅ APK generado: $apk"
  else
    log "⚠️ Build finalizado, pero no encontré APK en ./bin"
  fi
}

log "Iniciando flujo automatizado de build Android"

if [[ $SKIP_INSTALL -eq 0 ]]; then
  install_system_deps_if_needed
  create_or_update_venv
else
  log "Saltando instalación de dependencias por --no-install"
  [[ -d "$VENV_DIR" ]] || {
    echo "❌ No existe el entorno virtual en $VENV_DIR y se indicó --no-install"
    exit 1
  }
fi

run_build
log "Proceso completado."
