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
pip install -r requirements.txt

if (-not (Test-Path .env)) {
    New-Item -ItemType File -Path .env | Out-Null
}

docker compose up -d postgres redis
Start-Sleep -Seconds 6

python -m compileall trading_system tests scripts
pytest -q

Write-Output 'Instalación completa. Ejecuta: .\.venv\Scripts\Activate.ps1; python trading_system/app/main.py'
