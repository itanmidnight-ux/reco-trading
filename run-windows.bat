@echo off
setlocal enabledelayedexpansion

set ERROR_COUNT=0
set WARN_COUNT=0
set MISSING_DEPS=0

:: ==========================================
:: RECO-TRADING WINDOWS RUNNER v5.0
:: Fixed & Optimized for Windows
:: ==========================================

echo.
echo ========================================
echo   Reco-Trading Windows Runner v5.0
echo   Fixed & Optimized
echo ========================================
echo.

:: ============================================
:: CHECK PYTHON
:: ============================================

echo [INFO] Checking Python installation...

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo [INFO] Please install Python 3.10+ from: https://www.python.org/downloads/
    echo [INFO] Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version 2>nul | findstr /R "3\.1[0-9]" >nul
if errorlevel 1 (
    echo [WARN] Python version may be incompatible. Recommended: Python 3.10+
)

echo [OK] Python found

:: ============================================
:: CHECK AND CREATE VIRTUAL ENVIRONMENT
:: ============================================

echo.
echo [INFO] Checking virtual environment...

set CREATE_VENV=false
set VENV_PATH=.venv

if not exist "%VENV_PATH%" (
    set CREATE_VENV=true
    echo [INFO] Virtual environment not found, creating...
)

if "!CREATE_VENV!"=="true" (
    python -m venv %VENV_PATH% >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo [INFO] Try running: python -m pip install --upgrade pip
        set ERROR_COUNT=1
        goto :error_exit
    )
    echo [OK] Virtual environment created
)

if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment files incomplete
    echo [INFO] Deleting and recreating...
    rmdir /s /q %VENV_PATH% 2>nul
    python -m venv %VENV_PATH% >nul 2>&1
    if errorlevel 1 (
        set ERROR_COUNT=1
        goto :error_exit
    )
)

call %VENV_PATH%\Scripts\activate.bat >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    set ERROR_COUNT=1
    goto :error_exit
)

echo [OK] Virtual environment activated

:: ============================================
:: UPGRADE PIP
:: ============================================

echo.
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip -q 2>nul
echo [OK] pip upgraded

:: ============================================
:: CHECK AND INSTALL DEPENDENCIES
:: ============================================

echo.
echo [INFO] Checking dependencies...

:: Core dependencies
set PACKAGES=ccxt pandas sqlalchemy fastapi pydantic numpy ta psutil rich python-dotenv pyyaml pydantic-settings aiosqlite asyncpg httpx websockets aiohttp aiofiles flask uvicorn

for %%p in (%PACKAGES%) do (
    python -c "import %%p" 2>nul
    if errorlevel 1 (
        echo [MISSING] %%p
        pip install %%p -q 2>nul
        if not errorlevel 1 (
            echo [OK] %%p installed
        ) else (
            echo [WARN] Failed to install %%p
            set /a MISSING_DEPS+=1
        )
    )
)

:: Install from requirements.txt if it exists
if exist "requirements.txt" (
    echo [INFO] Installing from requirements.txt...
    pip install -r requirements.txt -q --ignore-installed 2>nul
    if errorlevel 1 (
        pip install -r requirements.txt 2>nul
    )
)

if %MISSING_DEPS% gtr 0 (
    echo [WARN] %MISSING_DEPS% dependencies had issues
) else (
    echo [OK] All dependencies installed
)

:: ============================================
:: CHECK .ENV FILE
:: ============================================

echo.
echo [INFO] Checking configuration...

if not exist ".env" (
    echo [WARN] .env file not found
    echo [INFO] Creating template .env file...
    
    (
        echo # Reco-Trading Configuration
        echo # Generated: %date% %time%
        echo.
        echo # Exchange API (REQUIRED - UPDATE THESE!)
        echo BINANCE_API_KEY=YOUR_API_KEY_HERE
        echo BINANCE_API_SECRET=YOUR_API_SECRET_HERE
        echo.
        echo # Trading Mode
        echo BINANCE_TESTNET=true
        echo CONFIRM_MAINNET=false
        echo ENVIRONMENT=testnet
        echo RUNTIME_PROFILE=paper
        echo.
        echo # Database (SQLite by default)
        echo DATABASE_URL=sqlite+aiosqlite:///./data/reco_trading.db
        echo.
        echo # Dashboard Security
        echo DASHBOARD_AUTH_ENABLED=true
        echo DASHBOARD_AUTH_MODE=token
        echo DASHBOARD_USERNAME=admin
        echo DASHBOARD_PASSWORD=admin
        echo DASHBOARD_API_TOKEN=CHANGE_THIS_TO_SECURE_TOKEN
        echo.
        echo # Feature Flags
        echo ENABLE_AUTO_IMPROVER=true
        echo ENABLE_ML_ENGINE=true
        echo ENABLE_CONTINUAL_LEARNING=true
    ) > .env
    
    echo [OK] Template .env created
    echo.
    echo [IMPORTANT] You MUST edit .env and add your Binance API keys!
    echo.
    pause
)

:: Check for API keys
findstr /C:"BINANCE_API_KEY=YOUR_API_KEY_HERE" .env >nul 2>&1
if not errorlevel 1 (
    echo [WARN] BINANCE_API_KEY not configured - using placeholder
)

findstr /C:"BINANCE_API_SECRET=YOUR_API_SECRET_HERE" .env >nul 2>&1
if not errorlevel 1 (
    echo [WARN] BINANCE_API_SECRET not configured - using placeholder
)

echo [OK] Configuration checked

:: ============================================
:: CHECK DIRECTORIES
:: ============================================

echo.
echo [INFO] Creating necessary directories...

if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "models" mkdir models

echo [OK] Directories ready

:: ============================================
:: VERIFY PROJECT FILES
:: ============================================

echo.
echo [INFO] Verifying project files...

if not exist "main.py" (
    if exist "reco_trading\main.py" (
        echo [INFO] Using reco_trading/main.py entry point
    ) else (
        echo [ERROR] No main.py or reco_trading/main.py found!
        echo [INFO] Make sure you're in the correct directory
        set ERROR_COUNT=1
        goto :error_exit
    )
)

if not exist "reco_trading" (
    echo [ERROR] reco_trading directory not found
    set ERROR_COUNT=1
    goto :error_exit
)

echo [OK] Project structure verified

:: ============================================
:: PARSE ARGUMENTS
:: ============================================

set RUN_MODE=default
set WEB_PORT=9000

:parse_args
if "%~1"=="" goto :run_start
if /I "%~1"=="--web" set RUN_MODE=web& shift& goto :parse_args
if /I "%~1"=="--api" set RUN_MODE=api& shift& goto :parse_args
if /I "%~1"=="--gui" set RUN_MODE=gui& shift& goto :parse_args
if /I "%~1"=="--test" set RUN_MODE=test& shift& goto :parse_args
if /I "%~1"=="--port" set WEB_PORT=%~2& shift& shift& goto :parse_args
if /I "%~1"=="--help" goto :show_help
shift
goto :parse_args

:show_help
echo.
echo Usage: run-windows.bat [OPTIONS]
echo.
echo Options:
echo   --web       Start Web Dashboard (default port 9000)
echo   --api       Start API Server (port 8080)
echo   --gui       Start GUI Application
echo   --test      Run tests
echo   --port N    Use port N for web/api
echo   --help      Show this help
echo.
exit /b 0

:run_start
echo.
echo [INFO] Starting Reco-Trading...
echo.

:: ============================================
:: RUN THE APPLICATION
:: ============================================

if "%RUN_MODE%"=="web" (
    echo [INFO] Mode: Web Dashboard (port %WEB_PORT%)
    echo [INFO] Dashboard will be available at http://localhost:%WEB_PORT%
    echo.
    python -m uvicorn reco_trading.api.main:app --host 0.0.0.0 --port %WEB_PORT% --reload
) else if "%RUN_MODE%"=="api" (
    echo [INFO] Mode: API Server
    python -m uvicorn reco_trading.api.main:app --host 0.0.0.0 --port %WEB_PORT%
) else if "%RUN_MODE%"=="gui" (
    echo [INFO] Mode: GUI Application
    python -m reco_trading.ui.app
) else if "%RUN_MODE%"=="test" (
    echo [INFO] Mode: Test
    python -m pytest -v
) else (
    echo [INFO] Mode: Default (Main Bot)
    echo.
    python main.py
)

:: ============================================
:: ERROR HANDLING
:: ============================================

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with errors
    echo.
    echo [INFO] Attempting safe mode...
    echo.
    
    python main.py --safe-mode 2>nul
    if not errorlevel 1 (
        echo [OK] Safe mode startup successful
    ) else (
        echo [ERROR] Safe mode also failed
        echo.
        echo [INFO] Common solutions:
        echo   1. Check .env file has correct API keys
        echo   2. Verify all dependencies installed
        echo   3. Check database connection
        echo   4. Review logs in ./logs directory
    )
)

goto :end

:error_exit
echo.
echo ========================================
echo   ERROR DURING STARTUP
echo ========================================
echo.
if %ERROR_COUNT% gtr 0 (
    echo [ERROR] %ERROR_COUNT% error(s) encountered
)
echo.
echo [INFO] Solutions:
echo   1. Install Python 3.10+ from python.org
echo   2. Run: install-windows.bat
echo   3. Configure .env with your API keys
echo   4. Ensure all dependencies are installed
echo.

:end
echo.
echo ========================================
echo   Execution Complete
echo ========================================
echo.

endlocal
pause
exit /b 0