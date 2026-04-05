@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set ERROR_COUNT=0
set WARN_COUNT=0

echo.
echo ========================================
echo   Reco-Trading Windows Installer v5.0
echo ========================================
echo.

for /f "tokens=2 delims==" %%A in ('wmic OS get TotalVisibleMemorySize /value 2^>nul ^| find "="') do set /a RAM_MB=%%A/1024
for /f "tokens=3" %%A in ('wmic logicaldisk where "DeviceID='C:'" get FreeSpace^,Size /format:value ^| find "FreeSpace"') do set FREE_SPACE=%%A
if not defined RAM_MB set RAM_MB=0
echo [INFO] RAM detectada: !RAM_MB! MB

if exist install-windows-powershell.ps1 (
    call :run_ps_installer powershell
    if !errorlevel! equ 0 goto :ok_exit
    call :run_ps_installer pwsh
    if !errorlevel! equ 0 goto :ok_exit
    echo [WARN] Los métodos PowerShell fallaron, usando instalador de emergencia CMD...
    set /a WARN_COUNT+=1
) else (
    echo [WARN] install-windows-powershell.ps1 no encontrado, usando instalador de emergencia CMD...
    set /a WARN_COUNT+=1
)

call :fallback_cmd_install
if !errorlevel! neq 0 (
    echo [ERROR] Instalación de emergencia falló.
    set /a ERROR_COUNT+=1
    goto :error_exit
)

goto :ok_exit

:run_ps_installer
set "PS_CMD=%~1"
where %PS_CMD% >nul 2>nul
if errorlevel 1 exit /b 1

echo [INFO] Ejecutando instalador con %PS_CMD%...
if /I "%PS_CMD%"=="powershell" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-windows-powershell.ps1"
) else (
    pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-windows-powershell.ps1"
)
exit /b %ERRORLEVEL%

:fallback_cmd_install
echo [INFO] Instalador CMD: verificando Python...
set PYTHON_CMD=
for %%P in (python py python3) do (
    where %%P >nul 2>nul && (
        set PYTHON_CMD=%%P
        goto :py_found
    )
)
:py_found
if not defined PYTHON_CMD (
    echo [ERROR] Python no encontrado. Instala Python 3.10+ y ejecuta de nuevo.
    exit /b 1
)

echo [OK] Python detectado: !PYTHON_CMD!
if not exist .venv (
    !PYTHON_CMD! -m venv .venv || exit /b 1
)
call .venv\Scripts\activate.bat || exit /b 1
python -m pip install --upgrade pip >nul 2>nul
if exist requirements.txt (
    python -m pip install -r requirements.txt || exit /b 1
)
if not exist data mkdir data
if not exist logs mkdir logs
if not exist .env (
    (
      echo BINANCE_API_KEY=CAMBIAR_POR_TU_API_KEY
      echo BINANCE_API_SECRET=CAMBIAR_POR_TU_API_SECRET
      echo BINANCE_TESTNET=true
      echo CONFIRM_MAINNET=false
      echo ENVIRONMENT=testnet
      echo RUNTIME_PROFILE=paper
      echo DATABASE_URL=sqlite+aiosqlite:///./data/reco_trading.db
      echo LLM_MODE=base
      echo LLM_REMOTE_ENDPOINT=
      echo LLM_REMOTE_MODEL=
      echo LLM_REMOTE_API_KEY=
      echo DASHBOARD_AUTH_ENABLED=true
      echo DASHBOARD_AUTH_MODE=token
      echo DASHBOARD_USERNAME=admin
      echo DASHBOARD_PASSWORD=admin
      echo DASHBOARD_API_TOKEN=CAMBIA_ESTE_TOKEN
    ) > .env
)
exit /b 0

:ok_exit
echo.
echo [OK] Instalación finalizada.
if %WARN_COUNT% gtr 0 echo [WARN] Advertencias: %WARN_COUNT%
echo.
pause
exit /b 0

:error_exit
echo.
echo [ERROR] Instalación finalizada con errores.
if %WARN_COUNT% gtr 0 echo [WARN] Advertencias: %WARN_COUNT%
echo.
pause
exit /b 1
