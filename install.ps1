$ErrorActionPreference = 'Stop'

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python no está instalado.'
}

python - <<'PY'
import sys
if sys.version_info < (3,11):
    raise SystemExit('Se requiere Python 3.11+')
print('Python OK')
PY

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

Write-Output 'Instalando dependencias basicas...'
pip install -r requirements.txt -q

# ============================================
# SMART ML DEPENDENCIES INSTALL
# ============================================

Write-Output 'Detectando recursos para ML optimizado...'

# Detect RAM (in GB)
$RAM_GB = [math]::Round((Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum / 1GB)

# Detect CPU cores
$CPU_CORES = (Get-CimInstance Win32_Processor).NumberOfCores

# Determine ML profile
if ($RAM_GB -ge 16 -and $CPU_CORES -ge 4) {
    $ML_PROFILE = "high"
    Write-Output "  Perfil ML: ALTO (RAM: ${RAM_GB}GB, CPU: $CPU_CORES)"
} elseif ($RAM_GB -ge 8 -and $CPU_CORES -ge 2) {
    $ML_PROFILE = "medium"
    Write-Output "  Perfil ML: MEDIO (RAM: ${RAM_GB}GB, CPU: $CPU_CORES)"
} else {
    $ML_PROFILE = "low"
    Write-Output "  Perfil ML: BAJO (RAM: ${RAM_GB}GB, CPU: $CPU_CORES) - optimizado"
}

# Helper function to check if package is installed
function Test-PythonPackage {
    param([string]$Package)
    try {
        python -c "import $Package" 2>$null
        return $true
    } catch {
        return $false
    }
}

Write-Output 'Instalando ML dependencies (perfil: {0})...' -f $ML_PROFILE

# PyTorch (core for all models)
if (-not (Test-PythonPackage "torch")) {
    Write-Output '  Instalando PyTorch (CPU)...'
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Output '  [OK] PyTorch instalado'
    } else {
        Write-Output '  [WARN] PyTorch installation failed, continuing'
    }
} else {
    Write-Output '  [OK] PyTorch ya instalado'
}

# PyTorch Lightning
if (-not (Test-PythonPackage "pytorch_lightning")) {
    pip install pytorch-lightning -q 2>$null
}

# LightGBM and XGBoost for existing FreqAI
if (-not (Test-PythonPackage "lightgbm")) {
    pip install lightgbm -q 2>$null
}

if (-not (Test-PythonPackage "xgboost")) {
    pip install xgboost -q 2>$null
}

# Advanced ML (only for medium/high profiles)
if ($ML_PROFILE -ne "low") {
    if (-not (Test-PythonPackage "stable_baselines3")) {
        pip install stable-baselines3 -q 2>$null
    }
    if (-not (Test-PythonPackage "optuna")) {
        pip install optuna -q 2>$null
    }
    if (-not (Test-PythonPackage "gymnasium")) {
        pip install gymnasium -q 2>$null
    }
}

# Light dependencies (all profiles)
if (-not (Test-PythonPackage "statsmodels")) {
    pip install statsmodels -q 2>$null
}

if (-not (Test-PythonPackage "yfinance")) {
    pip install yfinance -q 2>$null
}

Write-Output 'ML dependencies instaladas'

if (-not (Test-Path .env)) {
    @"
# RECO TRADING - AUTO-GENERATED CONFIG
# Generated: $(Get-Date)

# Exchange API
BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY
BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET

# Mode
BINANCE_TESTNET=true
CONFIRM_MAINNET=false
ENVIRONMENT=testnet
RUNTIME_PROFILE=paper

# Database (SQLite - works without admin)
DATABASE_URL=sqlite:///./data/reco_trading.db

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
"@ | Set-Content -Path .env -Encoding UTF8
}

docker compose up -d postgres redis
Start-Sleep -Seconds 6

python -m compileall reco_trading tests main.py
pytest -q

Write-Output 'Instalación completa. Ejecuta: .\.venv\Scripts\Activate.ps1; python main.py'
