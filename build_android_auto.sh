#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${APP_DIR:-$SCRIPT_DIR/app_android}"
VENV_DIR="${VENV_DIR:-$APP_DIR/.venv_android}"
STATE_FILE="${STATE_FILE:-$APP_DIR/.android_build_state}"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
JAVA_INSTALL_DIR="${JAVA_INSTALL_DIR:-$HOME/.java}"
JAVA_HOME_TARGET="${JAVA_HOME_TARGET:-$JAVA_INSTALL_DIR/jdk-17}"
JDK17_URL="${JDK17_URL:-https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse}"
FORCE_CLEAN=0
SKIP_INSTALL=0
JAVA_REINSTALLED=0

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

fail() {
  echo "❌ $*" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

upsert_env_var() {
  local file="$1"
  local key="$2"
  local value="$3"

  touch "$file"

  if grep -qE "^${key}=" "$file"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$file"
  else
    printf '%s=%s\n' "$key" "$value" >> "$file"
  fi
}

run_as_root() {
  if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    "$@"
  elif have_cmd sudo; then
    sudo "$@"
  else
    fail "Se requieren permisos de root para instalar paquetes y no se encontró sudo."
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
      fail "Gestor de paquetes no soportado: $pm"
      ;;
  esac
}

java_major_version() {
  local out
  out="$(java -version 2>&1 | head -n1 || true)"
  if [[ "$out" =~ \"([0-9]+)(\.[0-9]+)? ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  echo ""
}

validate_java17_ready() {
  local jv cv
  jv="$(java -version 2>&1 | tr '\n' ' ' || true)"
  cv="$(javac -version 2>&1 || true)"

  if [[ "$jv" != *"17"* ]]; then
    fail "java -version no reporta Java 17. Salida: $jv"
  fi

  if [[ "$cv" != *"17"* ]]; then
    fail "javac -version no reporta Java 17. Salida: $cv"
  fi
}

install_java17_manually() {
  log "Java 17 not found → installing"
  JAVA_REINSTALLED=1

  local sdkman_init="$HOME/.sdkman/bin/sdkman-init.sh"
  local sdk_java_candidate=""
  local sdk_java_home=""

  # Paso 1: limpiar JDK roto (Kali/Debian). Opcional, pero recomendado.
  if have_cmd apt-get; then
    log "Paso 1/9: limpiando OpenJDK previo con apt (si existe)"
    run_as_root apt-get remove --purge 'openjdk-*' -y || true
    run_as_root apt-get autoremove -y || true
  fi

  # Paso 2: instalar SDKMAN.
  if [[ ! -s "$sdkman_init" ]]; then
    log "Paso 2/9: instalando SDKMAN"
    have_cmd curl || fail "curl es requerido para instalar SDKMAN."
    curl -s "https://get.sdkman.io" | bash || fail "Falló instalación de SDKMAN."
  fi

  [[ -s "$sdkman_init" ]] || fail "SDKMAN no quedó instalado correctamente."

  # shellcheck disable=SC1090
  source "$sdkman_init"
  have_cmd sdk || fail "El comando sdk no está disponible tras inicializar SDKMAN."

  # Paso 3: listar versiones.
  log "Paso 3/9: listando versiones Java disponibles en SDKMAN"
  local sdk_list_output
  sdk_list_output="$(sdk list java || true)"
  [[ -n "$sdk_list_output" ]] || fail "No se pudo obtener la lista de Java desde SDKMAN."

  # Paso 4: instalar Java 17 (17.0.8-tem preferido, fallback a cualquier 17.x).
  log "Paso 4/9: instalando Java 17"
  if grep -q '17.0.8-tem' <<< "$sdk_list_output"; then
    sdk_java_candidate="17.0.8-tem"
  else
    sdk_java_candidate="$(awk '/\|/ && $0 ~ /17\./ && ($0 ~ /tem/ || $0 ~ /zulu/ || $0 ~ /amzn/ || $0 ~ /librca/) {gsub(/^[ \t]+|[ \t]+$/, "", $NF); print $NF; exit}' <<< "$sdk_list_output")"
  fi

  if [[ -z "$sdk_java_candidate" ]]; then
    log "SDKMAN no devolvió candidato Java 17 utilizable; usando fallback binario directo."

    mkdir -p "$JAVA_INSTALL_DIR"
    local tmp_dir archive extracted_dir
    tmp_dir="$(mktemp -d)"
    archive="$tmp_dir/jdk17.tar.gz"

    log "Downloading JDK..."
    if have_cmd curl; then
      curl -fL --retry 3 --connect-timeout 20 -o "$archive" "$JDK17_URL" || fail "Falló la descarga de JDK con curl."
    elif have_cmd wget; then
      wget -O "$archive" "$JDK17_URL" || fail "Falló la descarga de JDK con wget."
    else
      fail "Ni curl ni wget están instalados para descargar JDK 17."
    fi

    [[ -s "$archive" ]] || fail "El archivo descargado de JDK está vacío o inválido."

    log "Extracting..."
    tar -xzf "$archive" -C "$JAVA_INSTALL_DIR" || fail "Falló la extracción del archivo JDK."

    extracted_dir="$(tar -tzf "$archive" 2>/dev/null | head -n1 | cut -d/ -f1)"
    [[ -n "$extracted_dir" ]] || fail "No se pudo detectar carpeta extraída del JDK."
    [[ -d "$JAVA_INSTALL_DIR/$extracted_dir" ]] || fail "La carpeta extraída esperada no existe: $JAVA_INSTALL_DIR/$extracted_dir"

    rm -rf "$JAVA_HOME_TARGET"
    mv "$JAVA_INSTALL_DIR/$extracted_dir" "$JAVA_HOME_TARGET" || fail "No se pudo mover JDK a $JAVA_HOME_TARGET"

    rm -rf "$tmp_dir"
    return
  fi

  sdk install java "$sdk_java_candidate" || fail "Falló la instalación de Java 17 con SDKMAN ($sdk_java_candidate)."

  # Paso 5: usar Java 17 y dejarlo default.
  log "Paso 5/9: activando Java 17 en SDKMAN ($sdk_java_candidate)"
  sdk use java "$sdk_java_candidate" || fail "No se pudo activar Java 17 con sdk use."
  sdk default java "$sdk_java_candidate" || fail "No se pudo fijar Java 17 como default en SDKMAN."

  # Paso 7: configurar JAVA_HOME desde SDKMAN.
  log "Paso 7/9: configurando JAVA_HOME desde SDKMAN"
  sdk_java_home="$(sdk home java "$sdk_java_candidate" 2>/dev/null || true)"
  [[ -n "$sdk_java_home" && -d "$sdk_java_home" ]] || fail "SDKMAN no devolvió un JAVA_HOME válido para $sdk_java_candidate"

  mkdir -p "$JAVA_INSTALL_DIR"
  rm -rf "$JAVA_HOME_TARGET"
  ln -s "$sdk_java_home" "$JAVA_HOME_TARGET" || fail "No se pudo enlazar JAVA_HOME_TARGET a instalación SDKMAN."

  if [[ -f "$HOME/.zshrc" ]] && ! grep -q 'sdk home java' "$HOME/.zshrc"; then
    echo "export JAVA_HOME=\$(sdk home java $sdk_java_candidate)" >> "$HOME/.zshrc"
  fi
}
set_java_home_and_validate() {
  export JAVA_HOME="$JAVA_HOME_TARGET"
  export PATH="$JAVA_HOME/bin:$PATH"
  upsert_env_var "$ENV_FILE" "JAVA_HOME" "$JAVA_HOME"

  log "JAVA_HOME set to $JAVA_HOME"

  [[ -x "$JAVA_HOME/bin/java" ]] || fail "No existe ejecutable java en $JAVA_HOME/bin/java"
  [[ -x "$JAVA_HOME/bin/javac" ]] || fail "No existe ejecutable javac en $JAVA_HOME/bin/javac"

  log "Paso 6/9: verificando java -version y javac -version"
  java -version
  javac -version
  validate_java17_ready

  log "Java 17 ready"
}

install_system_deps_if_needed() {
  local pm
  pm="$(detect_pkg_manager)" || fail "No pude detectar gestor de paquetes. Instala manualmente Python 3 y herramientas de compilación."

  log "Gestor de paquetes detectado: $pm"

  local cmd_deps=(git zip unzip python3 pip3 curl wget)
  local missing_cmds=()
  for dep in "${cmd_deps[@]}"; do
    if ! have_cmd "$dep"; then
      missing_cmds+=("$dep")
    fi
  done

  if [[ ${#missing_cmds[@]} -gt 0 ]]; then
    log "Faltan comandos: ${missing_cmds[*]}"

    case "$pm" in
      apt-get)
        install_packages "$pm" \
          build-essential git zip unzip curl wget \
          python3 python3-venv python3-pip python3-dev \
          libffi-dev libssl-dev libsqlite3-dev zlib1g-dev
        ;;
      dnf|yum)
        install_packages "$pm" \
          gcc gcc-c++ make git zip unzip curl wget \
          python3 python3-pip python3-devel \
          libffi-devel openssl-devel sqlite-devel zlib-devel
        ;;
      pacman)
        install_packages "$pm" \
          base-devel git zip unzip curl wget \
          python python-pip
        ;;
      zypper)
        install_packages "$pm" \
          -t pattern devel_basis
        install_packages "$pm" \
          git zip unzip curl wget \
          python3 python3-pip python3-devel libffi-devel libopenssl-devel
        ;;
      apk)
        install_packages "$pm" \
          build-base git zip unzip curl wget \
          python3 py3-pip python3-dev libffi-dev openssl-dev
        ;;
    esac
  else
    log "Dependencias base ya instaladas, no se reinstalan."
  fi

  if [[ -x "$JAVA_HOME_TARGET/bin/java" ]] && [[ "$($JAVA_HOME_TARGET/bin/java -version 2>&1 | head -n1)" == *"17"* ]]; then
    log "Java 17 ya instalado, reutilizando $JAVA_HOME_TARGET"
  elif have_cmd java && [[ "$(java_major_version)" == "17" ]]; then
    log "Java 17 detectado en sistema, pero se normaliza en $JAVA_HOME_TARGET para build reproducible"
    install_java17_manually
  else
    install_java17_manually
  fi

  set_java_home_and_validate
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
    [[ -f "$APP_DIR/$f" ]] && existing+=("$APP_DIR/$f")
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
  [[ $JAVA_REINSTALLED -eq 1 ]] && return 0
  [[ ! -d "$APP_DIR/.buildozer" ]] && return 0

  local current_hash old_hash
  current_hash="$(hash_build_inputs)"
  old_hash=""
  [[ -f "$STATE_FILE" ]] && old_hash="$(cat "$STATE_FILE")"

  [[ "$current_hash" != "$old_hash" ]]
}

run_build() {
  [[ -f "$APP_DIR/buildozer.spec" ]] || fail "No se encontró buildozer.spec en $APP_DIR"

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  # Forzar Java 17 para Buildozer, independientemente del Java del sistema.
  export JAVA_HOME="$JAVA_HOME_TARGET"
  export PATH="$JAVA_HOME/bin:$PATH"

  cd "$APP_DIR"

  if needs_clean_build; then
    log "Paso 8/9: limpieza Buildozer (clean + regeneración)"
    log "Se detectaron cambios de configuración o no existe build previo. Recompilación completa..."
    buildozer android clean
  else
    log "Sin cambios estructurales. Build incremental..."
  fi

  log "Paso 9/9: compilando APK (debug)"
  buildozer -v android debug

  hash_build_inputs > "$STATE_FILE"

  local apk
  apk="$(find bin -maxdepth 1 -type f -name '*.apk' | sort | tail -n 1 || true)"
  if [[ -n "$apk" ]]; then
    log "✅ APK generado: $APP_DIR/$apk"
  else
    log "⚠️ Build finalizado, pero no encontré APK en $APP_DIR/bin"
  fi
}

log "Iniciando flujo automatizado de build Android"

if [[ $SKIP_INSTALL -eq 0 ]]; then
  install_system_deps_if_needed
  create_or_update_venv
else
  log "Saltando instalación de dependencias por --no-install"
  [[ -d "$VENV_DIR" ]] || fail "No existe el entorno virtual en $VENV_DIR y se indicó --no-install"
  set_java_home_and_validate
fi

run_build
log "Proceso completado."
