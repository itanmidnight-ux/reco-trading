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
echo [INFO] Instalando dependencias...
python -m pip install -r requirements.txt -q 2>nul
echo [OK] Dependencias instaladas

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
    echo [ERROR] Error en imports
) else (
    echo [OK] Modulos verificados
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