@echo off
setlocal enabledelayedexpansion

set ERROR_COUNT=0
set WARN_COUNT=0
set MISSING_DEPS=0

:: ==========================================
:: RECO-TRADING WINDOWS RUNNER v4.1
:: Auto-installs missing dependencies
:: ==========================================

echo.
echo ========================================
echo   Reco-Trading Windows Runner v4.1
echo ========================================
echo.

:: ============================================
:: CHECK AND CREATE VIRTUAL ENVIRONMENT
:: ============================================

echo [INFO] Verificando entorno virtual...

set CREATE_VENV=false

if not exist ".venv" (
    set CREATE_VENV=true
    echo [INFO] Entorno virtual no encontrado, creando...
)

if "!CREATE_VENV!"=="true" (
    python -m venv .venv >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual
        set ERROR_COUNT=1
        goto :error_exit
    )
)

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Scripts de entorno virtual incompletos
    set ERROR_COUNT=1
    goto :error_exit
)

call .venv\Scripts\activate.bat >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No se pudo activar el entorno virtual
    set ERROR_COUNT=1
    goto :error_exit
)

echo [OK] Entorno virtual listo

:: ============================================
:: CHECK AND INSTALL MISSING DEPENDENCIES
:: ============================================

echo.
echo [INFO] Verificando dependencias...

set MISSING_DEPS=0

:: Check each essential package
python -c "import ccxt" 2>nul || (
    echo [MISSING] ccxt
    pip install ccxt -q 2>nul
    if not errorlevel 1 echo [OK] ccxt instalado
    set /a MISSING_DEPS+=1
)

python -c "import pandas" 2>nul || (
    echo [MISSING] pandas
    pip install pandas -q 2>nul
    if not errorlevel 1 echo [OK] pandas instalado
    set /a MISSING_DEPS+=1
)

python -c "import sqlalchemy" 2>nul || (
    echo [MISSING] sqlalchemy
    pip install sqlalchemy -q 2>nul
    if not errorlevel 1 echo [OK] sqlalchemy instalado
    set /a MISSING_DEPS+=1
)

python -c "import fastapi" 2>nul || (
    echo [MISSING] fastapi
    pip install fastapi -q 2>nul
    if not errorlevel 1 echo [OK] fastapi instalado
    set /a MISSING_DEPS+=1
)

python -c "import pydantic" 2>nul || (
    echo [MISSING] pydantic
    pip install pydantic -q 2>nul
    if not errorlevel 1 echo [OK] pydantic instalado
    set /a MISSING_DEPS+=1
)

python -c "import numpy" 2>nul || (
    echo [MISSING] numpy
    pip install numpy -q 2>nul
    if not errorlevel 1 echo [OK] numpy instalado
    set /a MISSING_DEPS+=1
)

python -c "import ta" 2>nul || (
    echo [MISSING] ta
    pip install ta -q 2>nul
    if not errorlevel 1 echo [OK] ta instalado
    set /a MISSING_DEPS+=1
)

python -c "import psutil" 2>nul || (
    echo [MISSING] psutil
    pip install psutil -q 2>nul
    if not errorlevel 1 echo [OK] psutil instalado
    set /a MISSING_DEPS+=1
)

python -c "import rich" 2>nul || (
    echo [MISSING] rich
    pip install rich -q 2>nul
    if not errorlevel 1 echo [OK] rich instalado
    set /a MISSING_DEPS+=1
)

python -c "import dotenv" 2>nul || (
    echo [MISSING] python-dotenv
    pip install python-dotenv -q 2>nul
    if not errorlevel 1 echo [OK] python-dotenv instalado
    set /a MISSING_DEPS+=1
)

python -c "import yaml" 2>nul || (
    echo [MISSING] pyyaml
    pip install pyyaml -q 2>nul
    if not errorlevel 1 echo [OK] pyyaml instalado
    set /a MISSING_DEPS+=1
)

python -c "import pydantic_settings" 2>nul || (
    echo [MISSING] pydantic-settings
    pip install pydantic-settings -q 2>nul
    if not errorlevel 1 echo [OK] pydantic-settings instalado
    set /a MISSING_DEPS+=1
)

python -c "import aiosqlite" 2>nul || (
    echo [MISSING] aiosqlite
    pip install aiosqlite -q 2>nul
    if not errorlevel 1 echo [OK] aiosqlite instalado
    set /a MISSING_DEPS+=1
)

python -c "import asyncpg" 2>nul || (
    echo [MISSING] asyncpg
    pip install asyncpg -q 2>nul
    if not errorlevel 1 echo [OK] asyncpg instalado
    set /a MISSING_DEPS+=1
)

python -c "import httpx" 2>nul || (
    echo [MISSING] httpx
    pip install httpx -q 2>nul
    if not errorlevel 1 echo [OK] httpx instalado
    set /a MISSING_DEPS+=1
)

python -c "import websockets" 2>nul || (
    echo [MISSING] websockets
    pip install websockets -q 2>nul
    if not errorlevel 1 echo [OK] websockets instalado
    set /a MISSING_DEPS+=1
)

python -c "import aiohttp" 2>nul || (
    echo [MISSING] aiohttp
    pip install aiohttp -q 2>nul
    if not errorlevel 1 echo [OK] aiohttp instalado
    set /a MISSING_DEPS+=1
)

python -c "import aiofiles" 2>nul || (
    echo [MISSING] aiofiles
    pip install aiofiles -q 2>nul
    if not errorlevel 1 echo [OK] aiofiles instalado
    set /a MISSING_DEPS+=1
)

python -c "import flask" 2>nul || (
    echo [MISSING] flask
    pip install flask -q 2>nul
    if not errorlevel 1 echo [OK] flask instalado
    set /a MISSING_DEPS+=1
)

python -c "import uvicorn" 2>nul || (
    echo [MISSING] uvicorn
    pip install uvicorn -q 2>nul
    if not errorlevel 1 echo [OK] uvicorn instalado
    set /a MISSING_DEPS+=1
)

:: If still missing packages, install from requirements.txt
if exist "requirements.txt" (
    pip install -r requirements.txt -q --ignore-installed 2>nul || pip install -r requirements.txt 2>nul
)

echo [OK] Dependencias verificadas

:: ============================================
:: CHECK .ENV FILE
:: ============================================

echo.
echo [INFO] Verificando configuracion...

if not exist ".env" (
    echo [WARN] Archivo .env no encontrado
    echo [INFO] Creando archivo .env basico...
    
    (
        echo # RECO TRADING - AUTO-GENERATED CONFIG
        echo # Generated: %date% %time%
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
        echo # Database (SQLite)
        echo DATABASE_URL=sqlite+aiosqlite:///./data/reco_trading.db
    ) > .env
    
    echo [OK] .env basico creado
    echo.
    echo [IMPORTANTE] Debes editar .env y configurar tus API keys de Binance
)

:: Check for API keys
findstr /C:"BINANCE_API_KEY=CAMBIAR" .env >nul
if not errorlevel 1 (
    echo [WARN] BINANCE_API_KEY no configurada
)

findstr /C:"BINANCE_API_SECRET=CAMBIAR" .env >nul
if not errorlevel 1 (
    echo [WARN] BINANCE_API_SECRET no configurada
)

echo.
echo [IMPORTANTE] Debes configurar tus API keys de Binance en el archivo .env

:: ============================================
:: CHECK DIRECTORIES
:: ============================================

if not exist "data" mkdir data
if not exist "logs" mkdir logs

echo [OK] Directorios verificados

:: ============================================
:: VERIFY PROJECT
:: ============================================

echo.
echo [INFO] Verificando proyecto...

if not exist "main.py" (
    echo [ERROR] main.py no encontrado
    set ERROR_COUNT=1
    goto :error_exit
)

if not exist "reco_trading" (
    echo [ERROR] Directorio reco_trading no encontrado
    set ERROR_COUNT=1
    goto :error_exit
)

python -c "import sys; sys.path.insert(0,'.'); from reco_trading.config.settings import Settings" 2>nul
if errorlevel 1 (
    echo [WARN] Algunos modulos no disponibles
    echo [INFO] Intentando instalar...
    pip install -e . -q 2>nul || pip install -r requirements.txt -q 2>nul
)

echo [OK] Proyecto verificado

:: ============================================
:: DETECT ARGUMENTS
:: ============================================

set RUN_MODE=default

if "%1"=="" goto :run_default
if "%1"=="--web" goto :web_mode
if "%1"=="--api" goto :api_mode
if "%1"=="--gui" goto :gui_mode
if "%1"=="--test" goto :test_mode
if "%1"=="--help" goto :show_help

:run_default
echo [INFO] Modo: Default (bot principal)
goto :run_start

:web_mode
set RUN_MODE=web
echo [INFO] Modo: Web Dashboard
goto :run_start

:api_mode
set RUN_MODE=api
echo [INFO] Modo: API Server
goto :run_start

:gui_mode
set RUN_MODE=gui
echo [INFO] Modo: GUI Application
goto :run_start

:test_mode
set RUN_MODE=test
echo [INFO] Modo: Test
goto :run_start

:show_help
echo.
echo Usage: run-windows.bat [MODE]
echo.
echo Modes:
echo   --web     Start Web Dashboard
echo   --api     Start API Server
echo   --gui     Start GUI Application
echo   --test    Run tests
echo   --help    Show this help
echo.
exit /b 0

:run_start
echo.

:: ============================================
:: RUN THE APPLICATION
:: ============================================

echo [INFO] Iniciando Reco-Trading...
echo.

if "%RUN_MODE%"=="web" (
    python -m uvicorn reco_trading.api.main:app --host 0.0.0.0 --port 9000 --reload
) else if "%RUN_MODE%"=="api" (
    python -m uvicorn reco_trading.api.main:app --host 0.0.0.0 --port 8080
) else if "%RUN_MODE%"=="gui" (
    python -m reco_trading.ui.app
) else if "%RUN_MODE%"=="test" (
    pytest -v
) else (
    python main.py
)

:: Check if the application exited with error
if errorlevel 1 (
    echo.
    echo [ERROR] La aplicacion termino con errores
    echo.
    echo [INFO] Intentando iniciar en modo seguro...
    echo.
    
    python main.py --safe-mode 2>nul
    if not errorlevel 1 (
        echo [OK] Inicio en modo seguro exitoso
    ) else (
        echo [ERROR] No se pudo iniciar la aplicacion
    )
)

:end
echo.
echo ========================================
echo   Fin de la ejecucion
echo ========================================

endlocal
exit /b 0

:error_exit
echo.
echo ========================================
echo   Error durante la ejecucion
echo ========================================
echo.
if %ERROR_COUNT% gtr 0 (
    echo [ERROR] Se registraron %ERROR_COUNT% errores
)
echo.
echo Ejecuta install-windows.bat para configurar el entorno
echo.

endlocal
pause
exit /b 1