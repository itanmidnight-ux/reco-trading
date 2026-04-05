#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

ERROR_COUNT=0
WARN_COUNT=0

handle_error() {
    ERROR_COUNT=$((ERROR_COUNT + 1))
    local msg="$1"
    log_error "$msg"
    echo "  El instalador continuará con otros componentes..."
}

handle_warning() {
    WARN_COUNT=$((WARN_COUNT + 1))
    local msg="$1"
    log_warn "$msg"
}

upsert_env_key() {
    local env_file="$1"
    local key="$2"
    local value="$3"

    if [[ ! -f "$env_file" ]]; then
        printf "%s=%s\n" "$key" "$value" > "$env_file"
        return
    fi

    if grep -q "^${key}=" "$env_file"; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$env_file"
    else
        printf "\n%s=%s\n" "$key" "$value" >> "$env_file"
    fi
}

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Reco-Trading Linux Installer v4.0${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# ============================================
# DETECT SYSTEM TYPE
# ============================================

log_info "Detectando sistema operativo..."

SYSTEM_TYPE=""
SYSTEM_NAME=""
IS_ANDROID=false
IS_TERMUX=false
IS_PI=false

if [[ -f /proc/version ]]; then
    if grep -qi "android" /proc/version 2>/dev/null; then
        IS_ANDROID=true
        SYSTEM_TYPE="android"
        SYSTEM_NAME="Android"
    fi
fi

if [[ -n "${TERMUX_VERSION:-}" ]] || [[ -d "/data/data/com.termux" ]]; then
    IS_TERMUX=true
    SYSTEM_TYPE="termux"
    SYSTEM_NAME="Termux (Android)"
fi

if [[ -f /etc/os-release ]]; then
    source /etc/os-release
    case "${ID:-}${ID_LIKE:-}" in
        *debian*|*ubuntu*|*linuxmint*|debian|ubuntu)
            SYSTEM_TYPE="debian"
            SYSTEM_NAME="${PRETTY_NAME:-Debian/Ubuntu}"
            ;;
        *fedora*|fedora)
            SYSTEM_TYPE="fedora"
            SYSTEM_NAME="${PRETTY_NAME:-Fedora}"
            ;;
        *rhel*|*centos*|rhel|centos)
            SYSTEM_TYPE="rhel"
            SYSTEM_NAME="${PRETTY_NAME:-RHEL/CentOS}"
            ;;
        *arch*|arch|manjaro)
            SYSTEM_TYPE="arch"
            SYSTEM_NAME="${PRETTY_NAME:-Arch Linux}"
            ;;
        *alpine*)
            SYSTEM_TYPE="alpine"
            SYSTEM_NAME="${PRETTY_NAME:-Alpine}"
            ;;
        *suse*|opensuse*)
            SYSTEM_TYPE="suse"
            SYSTEM_NAME="${PRETTY_NAME:-openSUSE/SUSE}"
            ;;
    esac
fi

if [[ -f /sys/firmware/devicetree/base/model ]]; then
    if grep -qi "raspberry" /sys/firmware/devicetree/base/model 2>/dev/null; then
        IS_PI=true
        SYSTEM_TYPE="pi"
        SYSTEM_NAME="Raspberry Pi"
    fi
fi

if [[ -z "$SYSTEM_TYPE" ]]; then
    SYSTEM_TYPE="unknown"
    SYSTEM_NAME="Linux Desconocido"
fi

echo "  Sistema detectado: ${SYSTEM_NAME}"
echo "  Tipo: ${SYSTEM_TYPE}"

# ============================================
# DETECT ROOT ACCESS
# ============================================

IS_ROOT=false
IS_SUDO=false

if [[ "${EUID}" -eq 0 ]]; then
    IS_ROOT=true
    log_info "Ejecutando como ROOT"
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    IS_SUDO=true
    log_info "SUDO disponible sin contraseña"
else
    log_warn "Sin acceso root/sudo - modo usuario"
fi

RUN_CMD=""
if [[ "$IS_ROOT" == "true" ]]; then
    RUN_CMD=""
elif [[ "$IS_SUDO" == "true" ]]; then
    RUN_CMD="sudo"
fi

# ============================================
# DETECT RESOURCES
# ============================================

log_info "Detectando recursos del sistema..."

TOTAL_MEM_KB=0
if [[ -f /proc/meminfo ]]; then
    TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
fi
TOTAL_MEM_GB=$((TOTAL_MEM_KB / 1024 / 1024))
if [[ $TOTAL_MEM_GB -eq 0 ]]; then
    TOTAL_MEM_GB=1
fi

CPU_CORES=1
if [[ -f /proc/cpuinfo ]]; then
    CPU_CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "1")
fi

DISK_FREE_GB=5
if [[ -f /proc/mounts ]]; then
    DISK_FREE_GB=$(df -BG / 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
fi

echo "  RAM: ${TOTAL_MEM_GB} GB"
echo "  CPU Cores: ${CPU_CORES}"
echo "  Disk libre: ${DISK_FREE_GB} GB"

if [[ $TOTAL_MEM_GB -lt 1 ]]; then
    handle_warning "Memoria muy baja (${TOTAL_MEM_GB}GB). Instalar puede fallar."
fi

# ============================================
# DETECT PYTHON
# ============================================

log_info "Buscando Python..."

PYTHON_CMD=""
PYTHON_VERSION=""

for cmd in python3.13 python3.12 python3.11 python3.10 python3 python python3.9 python3.8; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON_CMD="$cmd"
        PYTHON_VERSION=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+' || echo "unknown")
        break
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    log_error "Python no encontrado"
    echo ""
    echo "Por favor instala Python 3.8+ manualmente:"
    echo "  Ubuntu/Debian: apt install python3 python3-venv python3-pip"
    echo "  Fedora: dnf install python3 python3-pip"
    echo "  Termux: pkg install python"
    echo "  Android (Termux): pkg install python"
    exit 1
fi

log_success "Python encontrado: ${PYTHON_VERSION} (${PYTHON_CMD})"

# ============================================
# INSTALL SYSTEM DEPENDENCIES
# ============================================

log_info "Instalando dependencias del sistema..."

install_package() {
    local pkg="$1"
    local cmd="$2"
    
    if command -v "$cmd" >/dev/null 2>&1; then
        return 0
    fi
    
    case "$SYSTEM_TYPE" in
        debian)
            export DEBIAN_FRONTEND=noninteractive
            ${RUN_CMD} apt-get update -qq 2>/dev/null || ${RUN_CMD} apt-get update 2>/dev/null || handle_warning "No se pudo actualizar repos"
            ${RUN_CMD} apt-get install -y -qq "$pkg" 2>/dev/null || ${RUN_CMD} apt-get install -y "$pkg" 2>/dev/null || handle_warning "No se pudo instalar $pkg"
            ;;
        fedora|rhel)
            ${RUN_CMD} dnf install -y "$pkg" 2>/dev/null || ${RUN_CMD} yum install -y "$pkg" 2>/dev/null || handle_warning "No se pudo instalar $pkg"
            ;;
        arch)
            ${RUN_CMD} pacman -Sy --noconfirm "$pkg" 2>/dev/null || handle_warning "No se pudo instalar $pkg"
            ;;
        termux)
            pkg install -y "$pkg" 2>/dev/null || handle_warning "No se pudo instalar $pkg"
            ;;
        alpine)
            ${RUN_CMD} apk add "$pkg" 2>/dev/null || handle_warning "No se pudo instalar $pkg"
            ;;
        suse)
            ${RUN_CMD} zypper --non-interactive install "$pkg" 2>/dev/null || handle_warning "No se pudo instalar $pkg"
            ;;
        pi)
            ${RUN_CMD} apt-get update -qq 2>/dev/null
            ${RUN_CMD} apt-get install -y -qq "$pkg" 2>/dev/null || handle_warning "No se pudo instalar $pkg"
            ;;
    esac
}

install_package "python3" "python3"
install_package "python3-venv" "python3"
install_package "python3-pip" "pip3"
install_package "curl" "curl"
install_package "git" "git"

# Verify pip
if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
    handle_warning "pip no disponible, intentando instalar..."
    
    case "$SYSTEM_TYPE" in
        debian)
            ${RUN_CMD} apt-get install -y -qq python3-pip 2>/dev/null || handle_error "No se pudo instalar pip"
            ;;
        termux)
            pip install -U pip 2>/dev/null || python -m ensurepip 2>/dev/null || handle_error "No se pudo instalar pip"
            ;;
    esac
fi

log_success "Dependencias del sistema verificadas"

# ============================================
# LLM MODE SELECTION
# ============================================

DASHBOARD_AUTH_ENABLED="true"
DASHBOARD_AUTH_MODE="token"
DASHBOARD_USERNAME="admin"
DASHBOARD_PASSWORD="admin"
DASHBOARD_API_TOKEN=""

if [[ -f ".env" ]]; then
    EXISTING_TOKEN=$(grep -E '^DASHBOARD_API_TOKEN=' .env | head -1 | cut -d '=' -f2- || true)
    if [[ -n "${EXISTING_TOKEN}" ]]; then
        DASHBOARD_API_TOKEN="${EXISTING_TOKEN}"
    fi
fi
if [[ -z "${DASHBOARD_API_TOKEN}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        DASHBOARD_API_TOKEN=$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(24))
PY
)
    else
        DASHBOARD_API_TOKEN="$(date +%s)_$RANDOM"
    fi
fi


# ============================================
# CREATE VIRTUAL ENVIRONMENT
# ============================================

log_info "Configurando entorno virtual..."

if [[ -d ".venv" ]]; then
    log_warn "Entorno virtual existente encontrado"
    echo ""
    read -p "¿Deseas eliminar el entorno virtual anterior y crear uno nuevo? (s/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        log_info "Eliminando entorno virtual anterior..."
        rm -rf .venv
    else
        log_info "Usando entorno virtual existente"
    fi
fi

if [[ ! -d ".venv" ]]; then
    $PYTHON_CMD -m venv .venv 2>/dev/null || handle_error "No se pudo crear el entorno virtual"
fi

if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f ".venv/Scripts/activate" ]]; then
    source .venv/Scripts/activate
fi

if command -v pip >/dev/null 2>&1; then
    pip install --upgrade pip -q --root-user-action=ignore 2>/dev/null || pip install --upgrade pip 2>/dev/null || handle_warning "No se pudo actualizar pip"
else
    $PYTHON_CMD -m pip install --upgrade pip 2>/dev/null || handle_warning "No se pudo usar pip"
fi

log_success "Entorno virtual creado"

# ============================================
# INSTALL PYTHON DEPENDENCIES
# ============================================

log_info "Instalando dependencias de Python..."

# Check if requirements.txt exists and install from it
if [[ -f "requirements.txt" ]]; then
    if command -v pip >/dev/null 2>&1; then
        # Upgrade pip first
        pip install --upgrade pip -q --root-user-action=ignore 2>/dev/null || \
        pip install --upgrade pip 2>/dev/null || true
        
        # Install from requirements.txt
        pip install -r requirements.txt -q --root-user-action=ignore 2>/dev/null || \
        pip install -r requirements.txt 2>/dev/null || \
        handle_warning "Algunas dependencias no se instalaron completamente"
        
        # Verify critical packages were installed
        CRITICAL_OK=true
        python -c "import ccxt" 2>/dev/null || CRITICAL_OK=false
        python -c "import pandas" 2>/dev/null || CRITICAL_OK=false
        python -c "import sqlalchemy" 2>/dev/null || CRITICAL_OK=false
        
        if [[ "$CRITICAL_OK" == "false" ]]; then
            handle_warning "Algunas dependencias críticas no se instalaron"
            log_info "Intentando instalar individualmente..."
            
            pip install ccxt pandas numpy sqlalchemy -q 2>/dev/null || true
        fi
    else
        $PYTHON_CMD -m pip install -r requirements.txt 2>/dev/null || \
        handle_warning "Error al instalar dependencias"
    fi
else
    handle_warning "requirements.txt no encontrado"
fi

log_success "Dependencias de Python instaladas"

# ============================================
# DETECT AND HANDLE DATABASE
# ============================================

log_info "Detectando base de datos..."

DB_TYPE="sqlite"
DB_EXISTS=false

if [[ -f "data/reco_trading.db" ]]; then
    DB_EXISTS=true
    log_warn "Base de datos SQLite existente encontrada"
fi

POSTGRES_AVAILABLE=false
MYSQL_AVAILABLE=false

if command -v pg_isready >/dev/null 2>&1; then
    if pg_isready -h localhost -p 5432 2>/dev/null; then
        POSTGRES_AVAILABLE=true
        DB_TYPE="postgresql"
        log_success "PostgreSQL detectado"
    fi
fi

if command -v mysql >/dev/null 2>&1; then
    if timeout 2 mysql -h localhost -e "SELECT 1" 2>/dev/null; then
        MYSQL_AVAILABLE=true
        DB_TYPE="mysql"
        log_success "MySQL detectado"
    fi
fi

if [[ "$DB_EXISTS" == "true" ]]; then
    echo ""
    log_warn "Se detectó una base de datos existente"
    echo ""
    echo "Opciones:"
    echo "  1) Mantener la base de datos actual"
    echo "  2) Eliminar y crear nueva base de datos"
    echo "  3) Crear backup y continuar"
    echo ""
    read -p "Elige una opción (1/2/3): " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            log_info "Mantendremos la base de datos existente"
            ;;
        2)
            log_info "Eliminando base de datos anterior..."
            rm -f data/reco_trading.db
            DB_EXISTS=false
            ;;
        3)
            if [[ -f "data/reco_trading.db" ]]; then
                BACKUP_FILE="data/reco_trading_backup_$(date +%Y%m%d_%H%M%S).db"
                cp data/reco_trading.db "$BACKUP_FILE"
                log_success "Backup creado: $BACKUP_FILE"
            fi
            rm -f data/reco_trading.db
            DB_EXISTS=false
            ;;
        *)
            log_info "Usando opción por defecto: mantener base de datos"
            ;;
    esac
fi

# ============================================
# HANDLE .ENV FILE
# ============================================

log_info "Configurando archivo .env..."

ENV_EXISTS=false
if [[ -f ".env" ]]; then
    ENV_EXISTS=true
    log_warn "Archivo .env existente encontrado"
    echo ""
    echo "Opciones:"
    echo "  1) Mantener .env actual"
    echo "  2) Crear backup y generar nuevo .env"
    echo "  3) Ver contenido actual"
    echo ""
    read -p "Elige una opción (1/2/3): " -n 1 -r
    echo ""
    
    case $REPLY in
        2)
            BACKUP_FILE=".env.backup_$(date +%Y%m%d_%H%M%S)"
            cp .env "$BACKUP_FILE"
            log_success "Backup creado: $BACKUP_FILE"
            ENV_EXISTS=false
            ;;
        3)
            echo "=== Contenido actual de .env ==="
            head -30 .env
            echo "================================"
            echo ""
            read -p "¿Deseas generar un nuevo .env? (s/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Ss]$ ]]; then
                BACKUP_FILE=".env.backup_$(date +%Y%m%d_%H%M%S)"
                cp .env "$BACKUP_FILE"
                log_success "Backup creado: $BACKUP_FILE"
                ENV_EXISTS=false
            fi
            ;;
        *)
            log_info "Mantendremos el .env actual"
            ;;
    esac
fi

if [[ "$ENV_EXISTS" == "false" ]]; then
    log_info "Generando archivo .env..."
    
    mkdir -p data logs
    
    if [[ "$DB_TYPE" == "postgresql" ]]; then
        cat > .env <<EOF
# ==============================
# RECO TRADING - AUTO-GENERATED CONFIG
# Generated: $(date)

# Exchange API (REEMPLAZA ESTAS CLAVES)
BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY
BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET

# Exchange Mode
BINANCE_TESTNET=true
CONFIRM_MAINNET=false
ENVIRONMENT=testnet
RUNTIME_PROFILE=paper

# Database (PostgreSQL)
POSTGRES_DSN=postgresql+asyncpg://trading:trading123@localhost:5432/reco_trading_prod
POSTGRES_ADMIN_DSN=postgresql+asyncpg://postgres@localhost:5432/postgres

# Redis
REDIS_URL=redis://localhost:6379/0

# AI/ML Configuration
ENABLE_AUTO_IMPROVER=true
ENABLE_ML_ENGINE=true
ENABLE_CONTINUAL_LEARNING=true
ENABLE_META_LEARNING=true
ENABLE_TFT=true
ENABLE_NBEATS=true
ENABLE_ADVANCED_META_LEARNING=true
ENABLE_REINFORCEMENT_LEARNING=true
DRIFT_DETECTION=true
ONCHAIN_ANALYSIS=true

# LLM Mode
# Usar valor por defecto interno (base)

# Dashboard Security
DASHBOARD_AUTH_ENABLED=${DASHBOARD_AUTH_ENABLED}
DASHBOARD_AUTH_MODE=${DASHBOARD_AUTH_MODE}
DASHBOARD_USERNAME=${DASHBOARD_USERNAME}
DASHBOARD_PASSWORD=${DASHBOARD_PASSWORD}
DASHBOARD_API_TOKEN=${DASHBOARD_API_TOKEN}
EOF
    elif [[ "$DB_TYPE" == "mysql" ]]; then
        cat > .env <<EOF
# ==============================
# RECO TRADING - AUTO-GENERATED CONFIG
# Generated: $(date)

# Exchange API
BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY
BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET

# Exchange Mode
BINANCE_TESTNET=true
CONFIRM_MAINNET=false
ENVIRONMENT=testnet
RUNTIME_PROFILE=paper

# Database (MySQL)
MYSQL_DSN=mysql+aiomysql://trading:trading123@localhost:3306/reco_trading_prod

# Redis
REDIS_URL=redis://localhost:6379/0

# AI/ML Configuration
ENABLE_AUTO_IMPROVER=true
ENABLE_ML_ENGINE=true
ENABLE_CONTINUAL_LEARNING=true
ENABLE_META_LEARNING=true
ENABLE_TFT=true
ENABLE_NBEATS=true
ENABLE_ADVANCED_META_LEARNING=true
ENABLE_REINFORCEMENT_LEARNING=true
DRIFT_DETECTION=true
ONCHAIN_ANALYSIS=true

# LLM Mode
# Usar valor por defecto interno (base)

# Dashboard Security
DASHBOARD_AUTH_ENABLED=${DASHBOARD_AUTH_ENABLED}
DASHBOARD_AUTH_MODE=${DASHBOARD_AUTH_MODE}
DASHBOARD_USERNAME=${DASHBOARD_USERNAME}
DASHBOARD_PASSWORD=${DASHBOARD_PASSWORD}
DASHBOARD_API_TOKEN=${DASHBOARD_API_TOKEN}
EOF
    else
        mkdir -p data
        cat > .env <<EOF
# ==============================
# RECO TRADING - AUTO-GENERATED CONFIG
# Generated: $(date)

# Exchange API
BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY
BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET

# Exchange Mode
BINANCE_TESTNET=true
CONFIRM_MAINNET=false
ENVIRONMENT=testnet
RUNTIME_PROFILE=paper

# Database (SQLite)
DATABASE_URL=sqlite+aiosqlite:///./data/reco_trading.db

# Redis (opcional)
REDIS_URL=redis://localhost:6379/0

# AI/ML Configuration
ENABLE_AUTO_IMPROVER=true
ENABLE_ML_ENGINE=true
ENABLE_CONTINUAL_LEARNING=true
ENABLE_META_LEARNING=true
ENABLE_TFT=true
ENABLE_NBEATS=true
ENABLE_ADVANCED_META_LEARNING=true
ENABLE_REINFORCEMENT_LEARNING=true
DRIFT_DETECTION=true
ONCHAIN_ANALYSIS=true

# LLM Mode
# Usar valor por defecto interno (base)

# Dashboard Security
DASHBOARD_AUTH_ENABLED=${DASHBOARD_AUTH_ENABLED}
DASHBOARD_AUTH_MODE=${DASHBOARD_AUTH_MODE}
DASHBOARD_USERNAME=${DASHBOARD_USERNAME}
DASHBOARD_PASSWORD=${DASHBOARD_PASSWORD}
DASHBOARD_API_TOKEN=${DASHBOARD_API_TOKEN}
EOF
    fi
    
    log_success ".env generado"
fi

upsert_env_key ".env" "DASHBOARD_AUTH_ENABLED" "${DASHBOARD_AUTH_ENABLED}"
upsert_env_key ".env" "DASHBOARD_AUTH_MODE" "${DASHBOARD_AUTH_MODE}"
upsert_env_key ".env" "DASHBOARD_USERNAME" "${DASHBOARD_USERNAME}"
upsert_env_key ".env" "DASHBOARD_PASSWORD" "${DASHBOARD_PASSWORD}"
upsert_env_key ".env" "DASHBOARD_API_TOKEN" "${DASHBOARD_API_TOKEN}"
log_success "Variables de decisión actualizadas en .env"

# ============================================
# CREATE DIRECTORIES
# ============================================

log_info "Creando estructura de directorios..."

mkdir -p data logs scripts/lib

if [[ ! -f "scripts/lib/runtime_env.sh" ]]; then
    cat > scripts/lib/runtime_env.sh <<'RUNTIME_ENV'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

load_dotenv_file() {
    local env_file="$1"
    if [[ -f "${env_file}" ]]; then
        set -a
        source "${env_file}"
        set +a
    fi
}

load_runtime_env() {
    cd "${ROOT_DIR}"
    load_dotenv_file .env
    load_dotenv_file config/database.env 2>/dev/null || true
}

load_runtime_env
RUNTIME_ENV
    chmod +x scripts/lib/runtime_env.sh
fi

log_success "Estructura de directorios creada"

# ============================================
# VERIFY INSTALLATION
# ============================================

log_info "Verificando instalación..."

if command -v python >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1; then
    TEST_PYTHON="${PYTHON_CMD:-python3}"
    
    $TEST_PYTHON -c "import sys; sys.path.insert(0,'.')" 2>/dev/null && \
        log_success "Python configurado correctamente" || \
        handle_warning "Error en verificación básica de Python"
fi

# ============================================
# FINAL OUTPUT
# ============================================

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ INSTALACIÓN COMPLETADA${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if [[ $ERROR_COUNT -gt 0 ]]; then
    echo -e "${YELLOW}⚠️  Se registraron ${ERROR_COUNT} errores durante la instalación${NC}"
fi
if [[ $WARN_COUNT -gt 0 ]]; then
    echo -e "${YELLOW}⚠️  Se registraron ${WARN_COUNT} advertencias${NC}"
fi

echo ""
echo -e "${YELLOW}⚠️  IMPORTANTE:${NC}"
echo "  Edita el archivo .env y reemplaza:"
echo "    - BINANCE_API_KEY"
echo "    - BINANCE_API_SECRET"
echo ""
echo -e "${CYAN}Para iniciar el bot:${NC}"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo ""
echo -e "${CYAN}Alternativamente:${NC}"
echo "  ./run.sh"
echo ""
