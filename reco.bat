@echo off
REM ==========================================
REM Reco-Trading Windows Launcher
REM ==========================================

echo ========================================
echo Reco-Trading Bot Launcher (Windows)
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if needed
if not exist "venv\Lib\site-packages\ccxt" (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Check for .env file
if not exist ".env" (
    echo WARNING: .env file not found
    echo Creating example .env file...
    (
        echo # Exchange API Keys
        echo BINANCE_API_KEY=your_api_key_here
        echo BINANCE_SECRET=your_secret_here
        echo.
        echo # Bot Configuration
        echo DRY_RUN=true
        echo LOG_LEVEL=INFO
    ) > .env.example
    echo Please copy .env.example to .env and add your API keys
    pause
)

REM Check for configuration
if not exist "config\settings.yaml" (
    echo Creating default config directory...
    mkdir config 2>nul
    echo Creating default settings...
    (
        echo # Reco-Trading Configuration
        echo trading:
        echo   symbol: BTC/USDT
        echo   timeframe: 5m
        echo   dry_run: true
        echo.
        echo autonomous:
        echo   enabled: true
        echo   decision_interval: 30
        echo   optimization_interval: 4
        echo.
        echo multipair:
        echo   enabled: true
        echo   pairs_count: 104
    ) > config\settings.yaml
)

REM Start the bot
echo.
echo ========================================
echo Starting Reco-Trading Bot...
echo ========================================
echo.

REM Parse command line arguments
set MODE=%1
if "%MODE%"=="" set MODE=run

if "%MODE%"=="run" (
    echo Running in LIVE mode...
    python -m reco_trading.main
) else if "%MODE%"=="dry" (
    echo Running in DRY-RUN mode...
    set DRY_RUN=true
    python -m reco_trading.main
) else if "%MODE%"=="backtest" (
    echo Running BACKTEST...
    python -m reco_trading.backtest %2 %3 %4
) else if "%MODE%"=="dashboard" (
    echo Starting WEB DASHBOARD on port 9000...
    python -m web_site.dashboard_server
) else (
    echo Unknown mode: %MODE%
    echo.
    echo Usage:
    echo   run       - Run bot in live mode
    echo   dry       - Run bot in dry-run mode
    echo   backtest  - Run backtesting
    echo   dashboard - Start web dashboard
    echo.
    echo Examples:
    echo   reco.bat run
    echo   reco.bat dry
    echo   reco.bat backtest --pairs BTC/USDT ETH/USDT --days 30
    echo   reco.bat dashboard
)

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Bot exited with error
    echo ========================================
    pause
)

REM Deactivate virtual environment
deactivate
