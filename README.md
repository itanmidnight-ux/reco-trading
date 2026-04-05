# Reco Trading

Bot de trading algorítmico para cripto con arquitectura asíncrona, motor de riesgo, dashboard web en tiempo real (SSE) y dashboard terminal (Rich TUI).

## Contenido
- [Características](#características)
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Instalación rápida](#instalación-rápida)
- [Configuración `.env`](#configuración-env)
- [Ejecución](#ejecución)
- [Dashboard web (seguridad)](#dashboard-web-seguridad)
- [Docker](#docker)
- [Operación y tuning de filtros](#operación-y-tuning-de-filtros)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Aviso importante](#aviso-importante)

---

## Características

- **Motor de trading asíncrono** con control de riesgo, sizing adaptativo y filtros por régimen.
- **Validación de señales multi-factor** (tendencia, momentum, volumen, estructura, volatilidad).
- **Adaptación automática de filtros** según actividad, calidad de señal y estado de sesión.
- **Soporte micro-balance** para cuentas pequeñas con ajuste dinámico de umbrales.
- **Dashboard web Flask** con endpoint SSE (`/api/stream`) y controles operativos.
- **Dashboard terminal Rich (TUI)** para operación headless en VPS/servidor.
- **Persistencia** SQLite/PostgreSQL/MySQL vía SQLAlchemy async.
- **Modo de decisión LLM configurable**: `base` o `llm_remote`.

---

## Arquitectura

```text
reco_trading/
├── core/               # Motor principal, estado, runtime settings, ejecución
├── strategy/           # Señales, filtros de régimen, confluencia
├── risk/               # Perfiles de capital, optimizer de inversión, sizing
├── database/           # Repository async y modelos
├── ui/                 # Dashboard terminal (Rich) + UI state
├── web_site/           # Dashboard web Flask + template SSE
└── api/                # API complementaria
```

---

## Requisitos

- Python **3.11+**
- Linux / Windows
- API Key y Secret de Binance (testnet recomendado para primeras pruebas)

Opcional:
- PostgreSQL/MySQL para persistencia avanzada
- Docker + Docker Compose

---

## Instalación rápida

### Linux

```bash
chmod +x install-linux.sh
./install-linux.sh
```

### Windows (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install-windows-powershell.ps1
```

> Los instaladores generan/actualizan `.env` y dejan el proyecto listo para ejecutar.
> También limpian variables obsoletas de integraciones antiguas (por ejemplo claves heredadas de Ollama).

---

## Configuración `.env`

Ejemplo base recomendado:

```env
# Exchange
BINANCE_API_KEY=TU_API_KEY
BINANCE_API_SECRET=TU_API_SECRET
BINANCE_TESTNET=true
CONFIRM_MAINNET=false

# Runtime
TRADING_SYMBOL=BTC/USDT
PRIMARY_TIMEFRAME=5m
CONFIRMATION_TIMEFRAME=15m

# Database (elige una)
DATABASE_URL=sqlite+aiosqlite:///./data/reco_trading.db
# POSTGRES_DSN=postgresql+asyncpg://user:pass@localhost:5432/reco_trading
# MYSQL_DSN=mysql+aiomysql://user:pass@localhost:3306/reco_trading

# LLM decision mode
LLM_MODE=base
LLM_REMOTE_ENDPOINT=https://api.openai.com/v1/chat/completions
LLM_REMOTE_MODEL=gpt-4o-mini
LLM_REMOTE_API_KEY=

# Dashboard security
DASHBOARD_AUTH_ENABLED=true
DASHBOARD_AUTH_MODE=token
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=admin
DASHBOARD_API_TOKEN=CAMBIA_ESTE_TOKEN
```

---

## Ejecución

### Entorno local

```bash
source .venv/bin/activate
python -m reco_trading.main
```

### Script helper

```bash
./run.sh
```

---

## Dashboard web (seguridad)

- URL local por defecto: `http://127.0.0.1:9000`
- Endpoints principales:
  - `/api/health`
  - `/api/snapshot`
  - `/api/stream` (SSE tiempo real)
  - `/api/settings` (runtime settings)
  - `/api/control/<action>`

Autenticación:
- `DASHBOARD_AUTH_MODE=token` (recomendado)
- `DASHBOARD_AUTH_MODE=basic`
- `DASHBOARD_AUTH_MODE=hybrid`

---

## Docker

Levantar stack:

```bash
docker compose up -d --build
```

Servicios incluidos:
- `app` (bot + dashboard web)
- `api` (API complementaria)
- `postgres`
- `redis`

Puertos expuestos en loopback (`127.0.0.1`) para reducir superficie de ataque.

---

## Operación y tuning de filtros

El bot usa configuración base por símbolo y ajuste adaptativo en runtime:

- Recalibración de filtros por datos recientes de mercado.
- Relax/tighten por actividad (`trades_today`), calidad (`signal_quality_score`) y riesgo.
- Refuerzo anti-falsas señales con controles sobre ADX/confianza/volumen.
- Modo micro-balance para mantener operatividad en cuentas pequeñas.

Recomendación operativa:
1. Empezar en **testnet**.
2. Observar 24–72h de telemetría (`autonomous_filters`, `autonomous_filter_reason`, `micro_balance_mode`).
3. Ajustar umbrales gradualmente (no cambios agresivos de una sola vez).

---

## Testing

Ejecutar suite principal:

```bash
pytest tests/ -x -q
```

Chequeos de sintaxis útiles:

```bash
python -m py_compile reco_trading/main.py web_site/dashboard_server.py reco_trading/core/bot_engine.py
bash -n run.sh install-linux.sh
```

---

## Troubleshooting

### 1) `Unauthorized` en web dashboard
- Verifica `DASHBOARD_AUTH_ENABLED`, `DASHBOARD_AUTH_MODE`, token/credenciales.

### 2) Pocos trades
- Revisa `min_confidence`, `adx_threshold`, `volume_buy_threshold`.
- Verifica `autonomous_filter_reason` y `micro_balance_mode` en snapshot.

### 3) Sin datos de mercado / señales neutras constantes
- Revisa conectividad con exchange y frescura de datos (`stale_market_data_ratio`).

### 4) Error DB
- Confirma DSN y disponibilidad del motor (SQLite/Postgres/MySQL).

---

## Aviso importante

Este software es educativo/técnico y **no constituye asesoramiento financiero**. Operar cripto implica alto riesgo. Usa testnet y gestión de riesgo estricta antes de operar en real.
