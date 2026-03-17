# Reco Trading Bot (Binance Spot)

Production-oriented algorithmic trading bot for small-capital spot trading on Binance, built around a state-machine engine, multi-factor voting signals, strict risk controls, and persistent telemetry.

## Overview

The bot runs with:

```bash
python main.py
```

It is compatible with:

```bash
./install.sh
./run.sh
```

Core goals:
- stable autonomous loop
- strong risk-first behavior
- Binance filter compliance (`LOT_SIZE`, `PRICE_FILTER`, `MIN_NOTIONAL`)
- persistent observability (trades/signals/market logs/state/errors)

---

## Architecture

```text
reco_trading/
  core/
    bot_engine.py          # Orchestration loop + cooldown + state transitions
    state_machine.py       # Bot states
  exchange/
    binance_client.py      # CCXT wrapper, retries, server-time sync
    order_manager.py       # Symbol rules, quantity/price normalization
  strategy/
    indicators.py          # EMA20/EMA50/RSI/ATR/volume MA
    signal_engine.py       # Multi-factor signal generation
    confidence_model.py    # Weighted voting confidence model
    regime_filter.py       # ATR/price volatility regime classifier
    order_flow.py          # Candle-based pressure estimator
  risk/
    risk_manager.py        # daily loss/trade count/confidence gates
    position_manager.py    # Position lifecycle and exits
  data/
    market_stream.py       # OHLCV fetch layer
    candle_builder.py      # DataFrame builder
  database/
    models.py              # SQLAlchemy models
    repository.py          # Async persistence APIs
  ui/
    dashboard.py           # Rich terminal dashboard
  config/
    settings.py            # Environment-backed runtime settings
main.py                    # Root entrypoint
```

---

## Trading Strategy

Signal voting combines:
- **Trend**: EMA20 vs EMA50 (5m + 15m confirmation)
- **Momentum**: RSI bias
- **Volume**: current volume vs MA20
- **Market structure**: higher-high/lower-low
- **Order flow**: buy/sell pressure from recent candle volume
- **Volatility regime gate**: ATR/price classification

### Volatility Regime Filter
`strategy/regime_filter.py` classifies market:
- `LOW_VOLATILITY`: no trade
- `NORMAL_VOLATILITY`: trade allowed
- `HIGH_VOLATILITY`: trade allowed with **30% size reduction**

### Order Flow System
`strategy/order_flow.py` computes buy pressure:
- `buy_pressure > 0.60` => `BUY`
- `buy_pressure < 0.40` => `SELL`
- otherwise => `NEUTRAL`

---

## Confidence Model

Weighted voting from all engines:
- trade allowed from `confidence >= 0.75`
- `>= 0.85`: strong
- `>= 0.90`: exceptional

---

## Risk Management Rules

- Max trades/day
- Daily loss limit (fraction of balance)
- Confidence threshold gate
- Cooldown after position close (**10 minutes**, configurable)
- Exchange filter validation before order placement

---

## Binance API Safety

- `load_markets()` and symbol-rule sync
- `fetch_time()` synchronization to reduce timestamp drift
- Retry/backoff on recoverable exchange/network failures
- Quantity rounding to `stepSize`
- Price rounding to `tickSize`
- Notional validation vs `minNotional`

---

## Terminal Dashboard (Rich)

Displays:
- bot status + state
- trading pair + timeframe + current price
- indicator signals + confidence grade
- volatility regime + order flow signal
- account and available balance
- open position details (entry/SL/TP/size)
- daily PnL

---

## Database Logging

Recorded entities:
- `trades`
- `signals`
- `market_data`
- `state_changes`
- `error_logs`
- `bot_logs`

All include timestamps and metadata for auditability.

---

## Installation and Run

### 1) Install
```bash
./install.sh
```

### 2) Configure environment variables
Required:
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `POSTGRES_DSN` (async SQLAlchemy DSN, e.g. `postgresql+asyncpg://...`)

Optional:
- `BINANCE_TESTNET` (`true`/`false`, default `true`)
- `ENVIRONMENT` (default `testnet`)
- `RUNTIME_PROFILE` (default `paper`)

### 3) Run
```bash
./run.sh
```
or directly:
```bash
python main.py
```

---

## Notes for Small Accounts

The bot is optimized for conservative operation:
- strict filters before trade entry
- reduced exposure in high volatility
- cooldown against overtrading
- daily risk cutoffs to prevent cascading losses

---

## Resiliencia 24/7 + API de control remoto

### Flujo de resiliencia implementado
- `reco_trading/main.py` ahora ejecuta el bot dentro de un supervisor global con reinicio automático.
- Se aplica **exponential backoff** con enfriamiento dinámico (`RESTART_BACKOFF_INITIAL_SECONDS` -> `RESTART_BACKOFF_MAX_SECONDS`).
- Se registra contador de reinicios y fallos consecutivos.
- Si se excede `MAX_CONSECUTIVE_FAILURES_BEFORE_PAUSE`, el sistema solicita pausa segura.
- Se añadió heartbeat periódico del `BotEngine` para evitar fallos silenciosos.

### API FastAPI segura
- Endpoints:
  - `GET /health`
  - `GET /metrics`
  - `GET /positions`
  - `POST /close-position`
  - `POST /pause`
  - `POST /resume`
  - `POST /kill-switch`
  - `POST /runtime-settings`
  - `POST /start`
- Seguridad por header:
  - `Authorization: Bearer <API_AUTH_KEY>`
- Variables requeridas:
  - `API_AUTH_KEY`
  - `BINANCE_API_KEY`
  - `BINANCE_API_SECRET` (o alias `BINANCE_SECRET`)

### systemd service
Archivo incluido: `bot.service`

Instalación:
```bash
sudo cp bot.service /etc/systemd/system/reco-trading.service
sudo systemctl daemon-reload
sudo systemctl enable reco-trading.service
sudo systemctl start reco-trading.service
sudo systemctl status reco-trading.service
```

Logs en vivo:
```bash
journalctl -u reco-trading.service -f
```


### Conexión móvil robusta (ngrok + auto-recuperación)
Para mantener la app Android conectada de forma estable y en tiempo real:

- `run.sh` ahora monitorea salud del túnel (`/public-url`) y reinicia ngrok si detecta caída.
- Actualiza automáticamente `PUBLIC_API_URL` en `.env` cuando cambia la URL del túnel.
- Soporta configuración estable de ngrok con:
  - `NGROK_AUTHTOKEN`
  - `NGROK_DOMAIN` (dominio reservado, recomendado)
  - `NGROK_REGION`
  - `NGROK_CHECK_INTERVAL_SECONDS`
  - `NGROK_MAX_RESTARTS_PER_HOUR`
- Soporta modo no interactivo con `RUN_MODE=1` (testnet) o `RUN_MODE=2` (mainnet).

Ejemplo:
```bash
RUN_MODE=1 NGROK_AUTHTOKEN=xxx NGROK_DOMAIN=tu-dominio.ngrok.app ./run.sh
```

> Recomendación operativa: usar dominio reservado de ngrok + authtoken para sesiones duraderas y reconexión más predecible en múltiples teléfonos.

### App Android (Buildozer)
Estructura en `app_android/` lista para compilar.

Compilación automática (recomendada):
```bash
cd app_android
./build_apk.sh
```

`build_apk.sh` ahora genera `app_android/runtime_config.json` antes de compilar, tomando datos de `.env` (como `PUBLIC_API_URL`, `RECO_API_URL`, `RECO_API_KEY`) o detectando URL de ngrok local si está activo. Así la APK queda lista para instalar en otro teléfono sin configuración manual posterior.

Comportamiento importante:
- **No instala OpenJDK por defecto** (ejecuta `build_android_auto.sh --no-install`).
- Valida que `java`/`javac` ya existan y sean Java 17.
- Falla rápido si faltan datos críticos (`RECO_API_URL`/`PUBLIC_API_URL` y `RECO_API_KEY` o `API_AUTH_KEY`).

Validación sin compilar (solo preparar configuración):
```bash
cd app_android
./build_apk.sh --prepare-only
```

Si quieres permitir instalación de dependencias del sistema de forma explícita:
```bash
cd app_android
./build_apk.sh --allow-install
```

Compilación manual:
```bash
cd app_android
pip install -r requirements.txt
buildozer -v android debug
```

APK generado en:
```text
app_android/bin/
```

### Notas de seguridad operativa
- No almacenar credenciales en código; usar `.env`.
- Rotar `API_AUTH_KEY` periódicamente.
- Exponer API solo por red privada/VPN o detrás de reverse proxy TLS.
- Usar `kill-switch` ante comportamiento anómalo del exchange o latencias extremas.

### Fallos posibles y mitigación
- **Fallo de exchange (ccxt/BaseError):** circuito de pausa temporal + reintentos automáticos.
- **Fallo inesperado de runtime:** supervisor reinicia automáticamente con backoff.
- **Credenciales faltantes:** validación en startup y bloqueo preventivo.
- **API no autorizada:** middleware rechaza token inválido con `401`.
