@echo off
setlocal enabledelayedexpansion

:: ==========================================
:: RECO-TRADING WINDOWS INSTALLER v3.0
:: Auto-detect and setup without admin if possible
:: ==========================================

echo.
echo ========================================
echo   Reco-Trading Windows Installer v3.0
echo ========================================
echo.

:: Check for Python
set PYTHON_CMD=
for %%P in (python python3 python3.11 python3.12 python3.13 py) do (
    where %%P >nul 2>nul && (
        set PYTHON_CMD=%%P
        goto :python_found
    )
)
:python_found

if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python no encontrado.
    echo.
    echo Instala Python 3.10+ desde:
    echo   https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [OK] Python: %PYTHON_CMD%

:: Check pip
%PYTHON_CMD% -m pip --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] pip no encontrado
    exit /b 1
)

:: Create virtual environment (works without admin)
echo.
echo [INFO] Creando entorno virtual...

if exist ".venv" (
    echo [WARN] Entorno existente, eliminando...
    rmdir /s /q ".venv" 2>nul
)

%PYTHON_CMD% -m venv .venv
call .venv\Scripts\activate.bat

echo [OK] Entorno virtual creado

:: Upgrade pip
python -m pip install --upgrade pip -q 2>nul

:: Install dependencies
echo [INFO] Instalando dependencias basicas...
python -m pip install -r requirements.txt -q 2>nul

:: ============================================
:: SMART ML DEPENDENCIES INSTALL
:: ============================================

echo [INFO] Detectando recursos para ML optimizado...

:: Detect RAM (approximate)
for /f "tokens=2 delims==" %%A in ('wmic OS get TotalVisibleMemorySize /value ^| find "="') do set TOTAL_MEM_KB=%%A
set /a ML_RAM_GB=%TOTAL_MEM_KB% / 1024 / 1024

:: Detect CPU cores
for /f %%A in ('wmic cpu get NumberOfCores ^| findstr [0-9]') do set ML_CPU_CORES=%%A

:: Determine ML profile
if %ML_RAM_GB% GEQ 16 (
    if %ML_CPU_CORES% GEQ 4 (
        set ML_PROFILE=high
        echo [INFO] Perfil ML: ALTO (RAM: %ML_RAM_GB%GB, CPU: %ML_CPU_CORES%)
    ) else (
        set ML_PROFILE=medium
        echo [INFO] Perfil ML: MEDIO (RAM: %ML_RAM_GB%GB, CPU: %ML_CPU_CORES%)
    )
) else if %ML_RAM_GB% GEQ 8 (
    if %ML_CPU_CORES% GEQ 2 (
        set ML_PROFILE=medium
        echo [INFO] Perfil ML: MEDIO (RAM: %ML_RAM_GB%GB, CPU: %ML_CPU_CORES%)
    ) else (
        set ML_PROFILE=low
        echo [INFO] Perfil ML: BAJO (RAM: %ML_RAM_GB%GB, CPU: %ML_CPU_CORES%) - optimizado
    )
) else (
    set ML_PROFILE=low
    echo [INFO] Perfil ML: BAJO (RAM: %ML_RAM_GB%GB, CPU: %ML_CPU_CORES%) - optimizado
)

:: Function to check if package is installed
echo [INFO] Instalando ML dependencies (perfil: %ML_PROFILE%)...

:: PyTorch (core for all models)
python -c "import torch" 2>nul
if errorlevel 1 (
    echo [INFO]   Instalando PyTorch (CPU)...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q 2>nul
    if not errorlevel 1 (
        echo [OK]   PyTorch instalado
    ) else (
        echo [WARN] PyTorch installation failed, continuing
    )
) else (
    echo [OK]   PyTorch ya instalado
)

:: PyTorch Lightning
python -c "import pytorch_lightning" 2>nul
if errorlevel 1 (
    python -m pip install pytorch-lightning -q 2>nul
)

:: LightGBM and XGBoost for existing FreqAI
python -c "import lightgbm" 2>nul
if errorlevel 1 (
    python -m pip install lightgbm -q 2>nul
)

python -c "import xgboost" 2>nul
if errorlevel 1 (
    python -m pip install xgboost -q 2>nul
)

:: Advanced ML (only for medium/high profiles)
if not "%ML_PROFILE%"=="low" (
    python -c "import stable_baselines3" 2>nul
    if errorlevel 1 (
        python -m pip install stable-baselines3 -q 2>nul
    )

    python -c "import optuna" 2>nul
    if errorlevel 1 (
        python -m pip install optuna -q 2>nul
    )

    python -c "import gymnasium" 2>nul
    if errorlevel 1 (
        python -m pip install gymnasium -q 2>nul
    )
)

:: Light dependencies (all profiles)
python -c "import statsmodels" 2>nul
if errorlevel 1 (
    python -m pip install statsmodels -q 2>nul
)

python -c "import yfinance" 2>nul
if errorlevel 1 (
    python -m pip install yfinance -q 2>nul
)

echo [OK] ML dependencies instaladas

:: Create directories
if not exist "data" mkdir data
if not exist "logs" mkdir logs

:: Auto-detect database
echo.
echo [INFO] Detectando base de datos...

:: Try to detect if PostgreSQL or MySQL are available
set DB_TYPE=sqlite

:: Check if psql is available
where psql >nul 2>nul
if not errorlevel 1 (
    :: Try to connect
    psql -h localhost -p 5432 -U postgres -c "SELECT 1" >nul 2>nul
    if not errorlevel 1 (
        set DB_TYPE=postgresql
    )
)

:: Check if mysql is available
if "%DB_TYPE%"=="sqlite" (
    where mysql >nul 2>nul
    if not errorlevel 1 (
        set DB_TYPE=mysql
    )
)

echo [INFO] Base de datos: %DB_TYPE%

:: Create .env
echo.
echo [INFO] Creando configuracion...

if exist ".env" (
    copy /Y .env .env.backup >nul
)

if "%DB_TYPE%"=="postgresql" (
    (
        echo # RECO TRADING - AUTO-GENERATED CONFIG
        echo # Generated: %date%
        echo.
        echo # Exchange API
        echo BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY
        echo BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET
        echo.
        echo # Mode
        echo BINANCE_TESTNET=true
        echo CONFIRM_MAINNET=false
        echo ENVIRONMENT=testnet
        echo RUNTIME_PROFILE=paper
        echo.
        echo # Database
        echo POSTGRES_DSN=postgresql+asyncpg://trading:trading123@localhost:5432/reco_trading_prod
        echo.
        echo # Redis
        echo REDIS_URL=redis://localhost:6379/0
        echo.
        echo # AI/ML Configuration
        echo ENABLE_AUTO_IMPROVER=true
        echo ENABLE_ML_ENGINE=true
        echo ENABLE_CONTINUAL_LEARNING=true
        echo ENABLE_META_LEARNING=true
        echo ENABLE_TFT=true
        echo ENABLE_NBEATS=true
        echo ENABLE_ADVANCED_META_LEARNING=true
        echo ENABLE_REINFORCEMENT_LEARNING=true
        echo DRIFT_DETECTION=true
        echo ONCHAIN_ANALYSIS=true
    ) > .env
    echo [OK] PostgreSQL configurado
) else if "%DB_TYPE%"=="mysql" (
    (
        echo # RECO TRADING - AUTO-GENERATED CONFIG
        echo # Generated: %date%
        echo.
        echo # Exchange API
        echo BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY
        echo BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET
        echo.
        echo # Mode
        echo BINANCE_TESTNET=true
        echo CONFIRM_MAINNET=false
        echo ENVIRONMENT=testnet
        echo RUNTIME_PROFILE=paper
        echo.
        echo # Database
        echo MYSQL_DSN=mysql+aiomysql://trading:trading123@localhost:3306/reco_trading_prod
        echo.
        echo # Redis
        echo REDIS_URL=redis://localhost:6379/0
        echo.
        echo # AI/ML Configuration
        echo ENABLE_AUTO_IMPROVER=true
        echo ENABLE_ML_ENGINE=true
        echo ENABLE_CONTINUAL_LEARNING=true
        echo ENABLE_META_LEARNING=true
        echo ENABLE_TFT=true
        echo ENABLE_NBEATS=true
        echo ENABLE_ADVANCED_META_LEARNING=true
        echo ENABLE_REINFORCEMENT_LEARNING=true
        echo DRIFT_DETECTION=true
        echo ONCHAIN_ANALYSIS=true
    ) > .env
    echo [OK] MySQL configurado
) else (
    :: SQLite fallback
    (
        echo # RECO TRADING - AUTO-GENERATED CONFIG
        echo # Generated: %date%
        echo.
        echo # Exchange API
        echo BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY
        echo BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET
        echo.
        echo # Mode
        echo BINANCE_TESTNET=true
        echo CONFIRM_MAINNET=false
        echo ENVIRONMENT=testnet
        echo RUNTIME_PROFILE=paper
        echo.
        echo # Database (SQLite - works without admin)
        echo DATABASE_URL=sqlite:///./data/reco_trading.db
        echo.
        echo # Redis
        echo REDIS_URL=redis://localhost:6379/0
        echo.
        echo # AI/ML Configuration
        echo ENABLE_AUTO_IMPROVER=true
        echo ENABLE_ML_ENGINE=true
        echo ENABLE_CONTINUAL_LEARNING=true
        echo ENABLE_META_LEARNING=true
        echo ENABLE_TFT=true
        echo ENABLE_NBEATS=true
        echo ENABLE_ADVANCED_META_LEARNING=true
        echo ENABLE_REINFORCEMENT_LEARNING=true
        echo DRIFT_DETECTION=true
        echo ONCHAIN_ANALYSIS=true
    ) > .env
    echo [OK] SQLite configurado (sin requerir servidor externo)
)

:: Create scripts directory
if not exist "scripts\lib" mkdir scripts\lib

:: Create runtime_env.sh stub
(
    echo @echo off
    echo setlocal enabledelayedexpansion
    echo if exist .env ^(
    echo     for /f "usebackq tokens=1,* delims==" %%%%a in ^(.env^) do ^(
    echo         set "%%%%a=%%%%b"
    echo     ^)
    echo ^)
) > scripts\lib\runtime_env.bat

echo [OK] Scripts de entorno creados

:: Verify
echo.
echo [INFO] Verificando instalacion...

python -c "import sys; sys.path.insert(0,'.'); from reco_trading.config.settings import Settings" 2>nul
if errorlevel 1 (
    echo [ERROR] Error en imports basicos
) else (
    echo [OK] Modulos basicos verificados
)

python -c "import sys; sys.path.insert(0,'.'); from reco_trading.ml.tft_model import TFTManager; from reco_trading.ml.nbeats_model import NBEATSManager; from reco_trading.ml.advanced_meta_learner import MetaLearningManager" 2>nul
if errorlevel 1 (
    echo [WARN] Algunos modulos ML no disponibles (verifica pip install)
) else (
    echo [OK] Modulos ML avanzados verificados
)

:: Final
echo.
echo ========================================
echo   INSTALACION COMPLETADA
echo ========================================
echo.
echo [IMPORTANTE]
echo   Edita el archivo .env y reemplaza:
echo   - BINANCE_API_KEY
echo   - BINANCE_API_SECRET
echo.
echo Para iniciar el bot:
echo   reco.bat
echo.

endlocal
pause