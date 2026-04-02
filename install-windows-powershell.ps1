# Reco-Trading Windows PowerShell Installer v4.0
# Professional installer with robust error handling

param(
    [switch]$SkipML,
    [switch]$Force,
    [switch]$NoConfirm
)

$ErrorActionPreference = 'Continue'
$Script:ErrorCount = 0
$Script:WarnCount = 0

function Write-Info { param($Msg) Write-Host "[INFO] $Msg" -ForegroundColor Cyan }
function Write-Success { param($Msg) Write-Host "[OK] $Msg" -ForegroundColor Green }
function Write-Warn { param($Msg); Write-Host "[WARN] $Msg" -ForegroundColor Yellow; $Script:WarnCount++ }
function Write-Error { param($Msg); Write-Host "[ERROR] $Msg" -ForegroundColor Red; $Script:ErrorCount++ }

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Reco-Trading Windows Installer v4.0" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ============================================
# DETECT SYSTEM
# ============================================

Write-Info "Detectando sistema..."

$IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Write-Host "  Admin: $(if($IsAdmin){'SI'}else{'NO'})"

$OSVersion = [System.Environment]::OSVersion.Version
Write-Host "  Windows: $($OSVersion.Major).$($OSVersion.Minor)"

# ============================================
# CHECK PYTHON
# ============================================

Write-Info "Buscando Python..."

$PythonCmd = $null
$PythonVersion = $null

$PythonCommands = @('python', 'python3', 'python3.13', 'python3.12', 'python3.11', 'python3.10', 'py')
foreach ($cmd in $PythonCommands) {
    try {
        $result = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0 -or $result -match "Python") {
            $PythonCmd = $cmd
            $PythonVersion = $result -replace "Python ", ""
            break
        }
    } catch { continue }
}

if (-not $PythonCmd) {
    Write-Error "Python no encontrado."
    Write-Host ""
    Write-Host "Por favor instala Python 3.8+ desde:"
    Write-Host "  https://www.python.org/downloads/"
    Write-Host ""
    exit 1
}

Write-Success "Python: $PythonVersion ($PythonCmd)"

# ============================================
# CHECK AND UPGRADE PIP
# ============================================

Write-Info "Verificando pip..."

try {
    $null = & $PythonCmd -m pip --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Warn "pip no disponible, intentando instalar..."
    try {
        & $PythonCmd -m ensurepip --upgrade 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "pip instalado"
        }
    } catch {
        Write-Error "No se pudo instalar pip"
    }
}

# Upgrade pip
& $PythonCmd -m pip install --upgrade pip -q 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Success "pip actualizado"
}

# ============================================
# VIRTUAL ENVIRONMENT
# ============================================

Write-Info "Configurando entorno virtual..."

if (Test-Path ".venv") {
    if ($NoConfirm) {
        Write-Info "Eliminando entorno virtual existente..."
        Remove-Item -Path ".venv" -Recurse -Force
    } else {
        Write-Warn "Entorno virtual existente encontrado"
        Write-Host "  1) Mantener entorno actual"
        Write-Host "  2) Eliminar y crear nuevo"
        $choice = Read-Host "Elige opción (1/2)"
        
        if ($choice -eq "2") {
            Write-Info "Eliminando entorno virtual..."
            Remove-Item -Path ".venv" -Recurse -Force
        }
    }
}

if (-not (Test-Path ".venv")) {
    & $PythonCmd -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "No se pudo crear el entorno virtual"
        exit 1
    }
}

# Activate venv
& .\.venv\Scripts\Activate.ps1 -ErrorAction Stop
Write-Success "Entorno virtual creado"

# ============================================
# INSTALL DEPENDENCIES
# ============================================

Write-Info "Instalando dependencias de Python..."

if (Test-Path "requirements.txt") {
    pip install -r requirements.txt -q 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Dependencias instaladas"
    } else {
        Write-Warn "Algunas dependencias no se instalaron correctamente"
    }
} else {
    Write-Warn "requirements.txt no encontrado"
}

# ============================================
# DETECT DATABASE
# ============================================

Write-Info "Detectando base de datos..."

$DBType = "sqlite"
$DBExists = $false

# Check existing SQLite
if (Test-Path "data\reco_trading.db") {
    $DBExists = $true
    Write-Warn "Base de datos existente encontrada"
}

# Check PostgreSQL
try {
    $psql = Get-Command psql -ErrorAction SilentlyContinue
    if ($psql) {
        $result = & psql -h localhost -p 5432 -U postgres -c "SELECT 1" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $DBType = "postgresql"
            Write-Success "PostgreSQL detectado"
        }
    }
} catch { }

# Check MySQL
if ($DBType -eq "sqlite") {
    try {
        $mysql = Get-Command mysql -ErrorAction SilentlyContinue
        if ($mysql) {
            $result = & mysql -h localhost -e "SELECT 1" 2>&1
            if ($LASTEXITCODE -eq 0) {
                $DBType = "mysql"
                Write-Success "MySQL detectado"
            }
        }
    } catch { }
}

Write-Host "  Base de datos: $DBType"

# Handle existing database
if ($DBExists -and -not $NoConfirm) {
    Write-Host ""
    Write-Warn "Se detectó una base de datos existente"
    Write-Host "  1) Mantener base de datos actual"
    Write-Host "  2) Eliminar y crear nueva"
    Write-Host "  3) Crear backup y continuar"
    $choice = Read-Host "Elige opción (1/2/3)"
    
    switch ($choice) {
        "2" {
            Remove-Item "data\reco_trading.db" -Force -ErrorAction SilentlyContinue
            Write-Info "Base de datos eliminada"
        }
        "3" {
            $backup = "data\reco_trading_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').db"
            Copy-Item "data\reco_trading.db" $backup -Force
            Write-Success "Backup creado: $backup"
            Remove-Item "data\reco_trading.db" -Force -ErrorAction SilentlyContinue
        }
    }
}

# ============================================
# HANDLE .ENV
# ============================================

Write-Info "Configurando archivo .env..."

$EnvExists = $false
if (Test-Path ".env") {
    $EnvExists = $true
    Write-Warn "Archivo .env existente encontrado"
    
    if (-not $NoConfirm) {
        Write-Host "  1) Mantener .env actual"
        Write-Host "  2) Crear backup y generar nuevo"
        Write-Host "  3) Ver contenido actual"
        $choice = Read-Host "Elige opción (1/2/3)"
        
        switch ($choice) {
            "2" {
                $backup = ".env.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
                Copy-Item ".env" $backup -Force
                Write-Success "Backup creado: $backup"
                $EnvExists = $false
            }
            "3" {
                Write-Host ""
                Write-Host "=== Contenido actual de .env ===" -ForegroundColor Yellow
                Get-Content ".env" | Select-Object -First 30
                Write-Host "==================================" -ForegroundColor Yellow
                Write-Host ""
                $rerun = Read-Host "Deseas generar un nuevo .env? (s/n)"
                if ($rerun -match "^[Ss]$") {
                    $backup = ".env.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
                    Copy-Item ".env" $backup -Force
                    Write-Success "Backup creado: $backup"
                    $EnvExists = $false
                }
            }
        }
    }
}

if (-not $EnvExists) {
    Write-Info "Generando archivo .env..."
    
    # Create directories
    New-Item -ItemType Directory -Path "data" -Force | Out-Null
    New-Item -ItemType Directory -Path "logs" -Force | Out-Null
    
    $envContent = @"
# ==============================
# RECO TRADING - AUTO-GENERATED CONFIG
# Generated: $(Get-Date)

# Exchange API
BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY
BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET

# Exchange Mode
BINANCE_TESTNET=true
CONFIRM_MAINNET=false
ENVIRONMENT=testnet
RUNTIME_PROFILE=paper

# Database ($DBType)
"@
    
    switch ($DBType) {
        "postgresql" {
            $envContent += @"

POSTGRES_DSN=postgresql+asyncpg://trading:trading123@localhost:5432/reco_trading_prod
"@
        }
        "mysql" {
            $envContent += @"

MYSQL_DSN=mysql+aiomysql://trading:trading123@localhost:3306/reco_trading_prod
"@
        }
        default {
            $envContent += @"

DATABASE_URL=sqlite+aiosqlite:///./data/reco_trading.db
"@
        }
    }
    
    $envContent += @"

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
"@
    
    $envContent | Set-Content -Path ".env" -Encoding UTF8
    
    if ($DBType -eq "sqlite") {
        Write-Success "SQLite configurado (sin permisos especiales)"
    } else {
        Write-Success "$DBType configurado"
    }
}

# ============================================
# CREATE DIRECTORIES
# ============================================

New-Item -ItemType Directory -Path "scripts\lib" -Force | Out-Null

# Create runtime_env.bat
$runtimeBat = @"
@echo off
setlocal enabledelayedexpansion
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%a in (.env) do (
        set "%%a=%%b"
    )
)
"@
Set-Content -Path "scripts\lib\runtime_env.bat" -Value $runtimeBat -Encoding UTF8

Write-Success "Estructura de directorios creada"

# ============================================
# VERIFY
# ============================================

Write-Info "Verificando instalación..."

try {
    python -c "import sys; sys.path.insert(0,'.')" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Python configurado correctamente"
    }
} catch {
    Write-Error "Error en verificación básica"
}

# ============================================
# FINAL OUTPUT
# ============================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  INSTALACIÓN COMPLETADA" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

if ($Script:ErrorCount -gt 0) {
    Write-Warn "Se registraron $($Script:ErrorCount) errores"
}

if ($Script:WarnCount -gt 0) {
    Write-Warn "Se registraron $($Script:WarnCount) advertencias"
}

Write-Host ""
Write-Host "[IMPORTANTE]" -ForegroundColor Yellow
Write-Host "  Edita el archivo .env y reemplaza:"
Write-Host "    - BINANCE_API_KEY"
Write-Host "    - BINANCE_API_SECRET"
Write-Host ""
Write-Host "Para iniciar el bot:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python main.py"
Write-Host ""
Write-Host "O simplemente:" -ForegroundColor Cyan
Write-Host "  .\reco.bat"
Write-Host ""
