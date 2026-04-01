@echo off
setlocal enabledelayedexpansion

:: ==========================================
:: RECO-TRADING WINDOWS LAUNCHER v3.0
:: Works without admin, auto-detects database
:: ==========================================

echo.
echo ========================================
echo   Reco-Trading Bot Launcher v3.0
echo ========================================
echo.

:: Check for virtual environment
if not exist ".venv" (
    echo [ERROR] Entorno virtual no encontrado
    echo Ejecuta install.bat primero
    pause
    exit /b 1
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado
    pause
    exit /b 1
)

echo [OK] Python: 

:: Load .env if exists
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (.env) do (
        set "%%a=%%b"
    )
)

:: Check API keys
if "%BINANCE_API_KEY%"=="" (
    echo [ERROR] BINANCE_API_KEY no configurada
    echo Edita .env y configura tus claves
    pause
    exit /b 1
)

if "%BINANCE_API_KEY%"=="CAMBIAR_POR_TU_API_KEY" (
    echo [WARN] Por favor configura tus API keys en .env
    echo   Reemplaza BINANCE_API_KEY con tu clave
    echo   Reemplaza BINANCE_API_SECRET con tu secreto
    pause
)

:: Verify database connection
echo.
echo [INFO] Verificando base de datos...

set DB_STATUS=unknown

:: Try PostgreSQL
if defined POSTGRES_DSN (
    python -c "import asyncio; import sys; sys.path.insert(0,'.'); from reco_trading.database.repository import Repository; asyncio.run(Repository('%POSTGRES_DSN%').verify_connectivity())" 2>nul
    if not errorlevel 1 (
        set DB_STATUS=PostgreSQL
        echo [OK] PostgreSQL conectado
        goto :db_check_done
    )
)

:: Try MySQL
if defined MYSQL_DSN (
    python -c "import asyncio; import sys; sys.path.insert(0,'.'); from reco_trading.database.repository import Repository; asyncio.run(Repository('%MYSQL_DSN%').verify_connectivity())" 2>nul
    if not errorlevel 1 (
        set DB_STATUS=MySQL
        echo [OK] MySQL conectado
        goto :db_check_done
    )
)

:: Fallback to SQLite
if "%DB_STATUS%"=="unknown" (
    if not defined DATABASE_URL (
        set DATABASE_URL=sqlite:///./data/reco_trading.db
        if not exist "data" mkdir data
        echo [INFO] Usando SQLite: %DATABASE_URL%
    )
    set DB_STATUS=SQLite
    echo [OK] SQLite configurado
) else if "%DB_STATUS%"=="PostgreSQL" (
    set DATABASE_URL=%POSTGRES_DSN%
) else if "%DB_STATUS%"=="MySQL" (
    set DATABASE_URL=%MYSQL_DSN%
)

:db_check_done

:: Menu
echo.
echo Selecciona modo de ejecucion:
echo   1) Testnet (Sandbox) - Recomendado
echo   2) Produccion Real (Dinero real)
echo   3) Dashboard Web
echo.
set /p MODE="Opcion [1]: "

if "%MODE%"=="1" (
    set BINANCE_TESTNET=true
    set ENVIRONMENT=testnet
    echo [OK] Modo Testnet
) else if "%MODE%"=="2" (
    echo [WARN] *** MODO PRODUCCION ***
    set /p CONFIRM="Escribe 'CONFIRMAR' para operar con dinero real: "
    if not "%CONFIRM%"=="CONFIRMAR" (
        echo Cancelado
        exit /b 1
    )
    set BINANCE_TESTNET=false
    set ENVIRONMENT=production
) else if "%MODE%"=="3" (
    echo Iniciando Dashboard Web...
    python -m web_site.dashboard_server
    exit /b 0
) else (
    set BINANCE_TESTNET=true
    set ENVIRONMENT=testnet
    echo [OK] Modo Testnet
)

:: Update .env with mode
if exist ".env" (
    powershell -Command "(Get-Content '.env') -replace 'BINANCE_TESTNET=.*', 'BINANCE_TESTNET=%BINANCE_TESTNET%' -replace 'ENVIRONMENT=.*', 'ENVIRONMENT=%ENVIRONMENT%' | Set-Content '.env'" 2>nul
)

:: Set non-interactive mode for Docker/automated execution
set DASHBOARD_TYPE=none

:: Start bot
echo.
echo ========================================
echo   Iniciando Reco-Trading Bot...
echo ========================================
echo.

python main.py

if errorlevel 1 (
    echo.
    echo [ERROR] El bot finalizo con error
)

pause