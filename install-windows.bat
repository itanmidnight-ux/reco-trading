@echo off
setlocal enabledelayedexpansion

set ERROR_COUNT=0
set WARN_COUNT=0

:: ==========================================
:: RECO-TRADING WINDOWS INSTALLER v4.0
:: Professional installer with error handling
:: ==========================================

echo.
echo ========================================
echo   Reco-Trading Windows Installer v4.0
echo ========================================
echo.

:: ============================================
:: DETECT SYSTEM
:: ============================================

echo [INFO] Detectando sistema...

:: Detect Windows version
for /f "tokens=2 delims=[]" %%A in ('ver') do set WIN_VERSION=%%A
for /f "tokens=1" %%A in ('echo %WIN_VERSION%') do set WIN_VERSION=%%A

echo   Windows: %WIN_VERSION%

:: Detect if running as admin
net session >nul 2>&1
if %errorLevel% equ 0 (
    set IS_ADMIN=true
    echo   Admin: SI
) else (
    set IS_ADMIN=false
    echo   Admin: NO
)

:: ============================================
:: CHECK PYTHON
:: ============================================

echo.
echo [INFO] Buscando Python...

set PYTHON_CMD=
set PYTHON_VERSION=

for %%P in (python python3 python3.13 python3.12 python3.11 python3.10 py) do (
    where %%P >nul 2>nul && (
        set PYTHON_CMD=%%P
        goto :python_found
    )
)

:python_found

if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python no encontrado.
    echo.
    echo Por favor instala Python 3.8+ desde:
    echo   https://www.python.org/downloads/
    echo.
    echo O usa el instalador de Microsoft Store:
    echo   python -m pip install --upgrade pip
    echo.
    set ERROR_COUNT=1
    pause
    exit /b 1
)

:: Get Python version
for /f "tokens=2" %%A in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%A
echo [OK] Python: %PYTHON_VERSION% (%PYTHON_CMD%)

:: ============================================
:: CHECK PIP
:: ============================================

%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip no disponible
    echo [INFO] Intentando instalar pip...
    %PYTHON_CMD% -m ensurepip --upgrade >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] No se pudo instalar pip
        set ERROR_COUNT=1
        pause
        exit /b 1
    )
)
echo [OK] pip disponible

:: ============================================
:: CREATE VIRTUAL ENVIRONMENT
:: ============================================

echo.
echo [INFO] Creando entorno virtual...

if exist ".venv" (
    echo [WARN] Entorno virtual existente encontrado
    echo.
    echo   1) Mantener entorno virtual actual
    echo   2) Eliminar y crear nuevo
    set /p VENV_CHOICE="Elige una opcion (1/2): "
    
    if "!VENV_CHOICE!"=="2" (
        echo [INFO] Eliminando entorno virtual anterior...
        rmdir /s /q ".venv" 2>nul
    ) else (
        echo [INFO] Usando entorno existente
        goto :venv_skip_create
    )
)

%PYTHON_CMD% -m venv .venv
if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual
    set ERROR_COUNT=1
    pause
    exit /b 1
)

:venv_skip_create

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] No se encontro script de activacion
    set ERROR_COUNT=1
    pause
    exit /b 1
)

echo [OK] Entorno virtual creado

:: ============================================
:: UPGRADE PIP
:: ============================================

python -m pip install --upgrade pip -q 2>nul
if errorlevel 1 (
    echo [WARN] No se pudo actualizar pip
) else (
    echo [OK] pip actualizado
)

:: ============================================
:: INSTALL DEPENDENCIES
:: ============================================

echo.
echo [INFO] Instalando dependencias...

if exist "requirements.txt" (
    python -m pip install -r requirements.txt -q 2>nul
    if errorlevel 1 (
        echo [WARN] Error al instalar algunas dependencias
        set /a WARN_COUNT+=1
    ) else (
        echo [OK] Dependencias instaladas
    )
) else (
    echo [WARN] requirements.txt no encontrado
    set /a WARN_COUNT+=1
)

:: ============================================
:: DETECT DATABASE
:: ============================================

echo.
echo [INFO] Detectando base de datos...

set DB_TYPE=sqlite
set DB_EXISTS=false

:: Check for existing SQLite database
if exist "data\reco_trading.db" (
    set DB_EXISTS=true
    echo [WARN] Base de datos existente encontrada
)

:: Check PostgreSQL
where psql >nul 2>nul
if not errorlevel 1 (
    psql -h localhost -p 5432 -U postgres -c "SELECT 1" >nul 2>&1
    if not errorlevel 1 (
        set DB_TYPE=postgresql
        echo [OK] PostgreSQL detectado
    )
)

:: Check MySQL
if "%DB_TYPE%"=="sqlite" (
    where mysql >nul 2>nul
    if not errorlevel 1 (
        mysql -h localhost -e "SELECT 1" >nul 2>&1
        if not errorlevel 1 (
            set DB_TYPE=mysql
            echo [OK] MySQL detectado
        )
    )
)

echo   Base de datos detectada: %DB_TYPE%

:: Handle existing database
if "%DB_EXISTS%"=="true" (
    echo.
    echo [WARN] Se detecto una base de datos existente
    echo.
    echo   1) Mantener la base de datos actual
    echo   2) Eliminar y crear nueva
    echo   3) Crear backup y continuar
    set /p DB_CHOICE="Elige una opcion (1/2/3): "
    
    if "!DB_CHOICE!"=="2" (
        echo [INFO] Eliminando base de datos anterior...
        del /f /q "data\reco_trading.db" 2>nul
        set DB_EXISTS=false
    ) else if "!DB_CHOICE!"=="3" (
        if exist "data\reco_trading.db" (
            set BACKUP_FILE=data\reco_trading_backup_%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%.db
            set BACKUP_FILE=!BACKUP_FILE: =0!
            copy /y "data\reco_trading.db" "!BACKUP_FILE!" >nul
            echo [OK] Backup creado: !BACKUP_FILE!
            del /f /q "data\reco_trading.db" 2>nul
            set DB_EXISTS=false
        )
    )
)

:: ============================================
:: HANDLE .ENV FILE
# ============================================

echo.
echo [INFO] Configurando archivo .env...

set ENV_EXISTS=false

if exist ".env" (
    set ENV_EXISTS=true
    echo [WARN] Archivo .env existente encontrado
    echo.
    echo   1) Mantener .env actual
    echo   2) Crear backup y generar nuevo
    echo   3) Ver contenido actual
    set /p ENV_CHOICE="Elige una opcion (1/2/3): "
    
    if "!ENV_CHOICE!"=="2" (
        set BACKUP_FILE=.env.backup_%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%.bat
        set BACKUP_FILE=!BACKUP_FILE: =0!
        copy /y ".env" "!BACKUP_FILE!" >nul
        echo [OK] Backup creado: !BACKUP_FILE!
        set ENV_EXISTS=false
    ) else if "!ENV_CHOICE!"=="3" (
        echo.
        echo === Contenido actual de .env ===
        type .env
        echo ==================================
        echo.
        set /p ENV_RERUN="Deseas generar un nuevo .env? (s/n): "
        if /i "!ENV_RERUN!"=="s" (
            set BACKUP_FILE=.env.backup_%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%.bat
            set BACKUP_FILE=!BACKUP_FILE: =0!
            copy /y ".env" "!BACKUP_FILE!" >nul
            echo [OK] Backup creado: !BACKUP_FILE!
            set ENV_EXISTS=false
        )
    )
)

if "%ENV_EXISTS%"=="false" (
    echo [INFO] Generando archivo .env...
    
    if not exist "data" mkdir data
    if not exist "logs" mkdir logs
    
    if "%DB_TYPE%"=="postgresql" (
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
        echo [OK] SQLite configurado (funciona sin permisos especiales)
    )
)

:: ============================================
:: CREATE DIRECTORIES
:: ============================================

if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "scripts\lib" mkdir scripts\lib

:: Create runtime_env.bat stub
(
    echo @echo off
    echo setlocal enabledelayedexpansion
    echo if exist .env ^(
    echo     for /f "usebackq tokens=1,* delims==" %%%%a in ^(.env^) do ^(
    echo         set "%%%%a=%%%%b"
    echo     ^)
    echo ^)
) > scripts\lib\runtime_env.bat

echo [OK] Estructura de directorios creada

:: ============================================
:: VERIFY INSTALLATION
:: ============================================

echo.
echo [INFO] Verificando instalacion...

python -c "import sys; sys.path.insert(0,'.')" 2>nul
if errorlevel 1 (
    echo [ERROR] Error en verificacion basica
    set /a ERROR_COUNT+=1
) else (
    echo [OK] Python configurado correctamente
)

:: ============================================
:: FINAL OUTPUT
:: ============================================

echo.
echo ========================================
echo   INSTALACION COMPLETADA
echo ========================================
echo.

if %ERROR_COUNT% gtr 0 (
    echo [WARN] Se registraron %ERROR_COUNT% errores
)

if %WARN_COUNT% gtr 0 (
    echo [WARN] Se registraron %WARN_COUNT% advertencias
)

echo.
echo [IMPORTANTE]
echo   Edita el archivo .env y reemplaza:
echo   - BINANCE_API_KEY
echo   - BINANCE_API_SECRET
echo.
echo Para iniciar el bot:
echo   call .venv\Scripts\activate.bat
echo   python main.py
echo.
echo O simplemente:
echo   reco.bat
echo.

endlocal
pause
