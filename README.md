# reco-trading (producción BTC/USDT Spot 5m)

Sistema cuantitativo modular para Binance Spot con activo único BTC/USDT en timeframe 5m.

## Arquitectura institucional objetivo

```text
                    ┌────────────────────────────┐
                    │        DATA FEED LAYER     │
                    │ WebSocket + REST Binance   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │     FEATURE ENGINE         │
                    │ Indicators + Microstruct   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ MARKET REGIME DETECTOR     │
                    │ HMM + GMM + Volatility     │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │   SIGNAL FUSION ENGINE     │
                    │ Probabilistic Ensemble     │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ INSTITUTIONAL RISK MANAGER │
                    │ Exposure + Correlation     │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ EXECUTION ENGINE (ASYNC)   │
                    │ Smart Order Router         │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ DATABASE + REDIS STATE     │
                    └────────────────────────────┘
```

## Estructura

- `reco_trading/config`: settings y logging.
- `reco_trading/core`: market data, features, modelos, fusión, riesgo, ejecución, portfolio.
- `reco_trading/infra`: cliente Binance, DB async y persistencia de estado en Redis.
- `reco_trading/research`: backtest profesional, walk-forward y Monte Carlo.
- `reco_trading/monitoring`: health checks y alertas.
- `database/schema.sql`: esquema SQL institucional para `reco_trading_prod`.
- `scripts/reset_database.sh`: reset completo de PostgreSQL con usuario `trading`.

## Instalación Debian 13

```bash
./install.sh
# install.sh sincroniza automáticamente POSTGRES_DSN en .env
# Edita únicamente tus API keys reales si aplica
./run.sh
```


## Estrategias permitidas en producción

El kernel valida una lista cerrada de estrategias habilitables por configuración:
- `directional` (stacking + transformer)
- `adaptive_market_making`
- `multi_exchange_arbitrage`

Los módulos experimentales quedan fuera del camino productivo y se ejecutan solo bajo perfil `research`.

## Seguridad

- Claves API en `.env` únicamente.
- `.env` ignorado por git.
- Usar API key sin permisos de retiro y con whitelist de IP.
- El sistema separa `BINANCE_TESTNET=false/true` por entorno.
- La inicialización de PostgreSQL es determinística mediante `config/database.env` + `install.sh` (idempotente).

## Solución de autenticación Postgres (usuario `trading`)

Si ves errores tipo `password authentication failed for user "trading"`:

```bash
./scripts/reset_database.sh
```

Y valida `pg_hba.conf` (ruta típica `/etc/postgresql/*/main/pg_hba.conf`) con:

```text
local   all   trading   md5
```

Luego reinicia:

```bash
sudo service postgresql restart
```


## Signal Fusion Engine institucional (implementado)

`reco_trading/core/signal_fusion.py` ahora incluye:
- memoria rolling de PnL por modelo,
- pesos dinámicos Sharpe con *recency decay* y *Bayesian shrinkage*,
- ajuste por régimen (`trend`, `range`, `volatile`),
- dampening por volatilidad,
- calibración online opcional (logistic/Platt-like) para probabilidad final.

## Reset DB robusto

El script `scripts/reset_database.sh` soporta variables de entorno operativas:
- `DB_NAME`, `DB_USER`, `DB_PASS`, `DB_HOST`, `DB_PORT`,
- `ADMIN_OS_USER` (default `root`, fallback automático a `postgres`),
- `SCHEMA_PATH`.

Ejemplo:

```bash
DB_PASS='trading_secure_2026' ADMIN_OS_USER=root ./scripts/reset_database.sh
```

## Despliegue operacional (systemd / docker compose)

Se añadió la carpeta `deploy/` con:
- unidades systemd para `orchestrator`, `trading-worker`, `evolution-worker`, `architecture-search-worker` y `self-healing-worker`.
- scripts de tuning Linux (`ulimit`, red kernel 6.x, afinidad CPU y cgroups).
- validación de compatibilidad de runtime para Python 3.11+, CUDA, Redis y PostgreSQL.
- perfiles para bare-metal y docker compose opcional.

Ver guía completa en `deploy/README.md`.


## Operación real: validaciones obligatorias Binance

El sistema arranca con validaciones estrictas para evitar estados degradados silenciosos:

- `symbol` obligatorio: `BTC/USDT`.
- En mainnet (`BINANCE_TESTNET=false`) exige `CONFIRM_MAINNET=true`.
- `BinanceClient.ping()` valida conexión real con `load_markets()` + `fetch_ticker('BTC/USDT')`.
- Si el precio inicial `last <= 0`, el kernel aborta.
- Se valida balance real (`fetch_balance`) al inicio y se registra en logs.
- En órdenes de mercado usa métodos explícitos de CCXT:
  - `create_market_buy_order('BTC/USDT', amount)`
  - `create_market_sell_order('BTC/USDT', amount)`

## Dashboard web local (métricas operativas)

La vista web local muestra en vivo:

- Capital
- Balance real Binance
- PnL diario
- Win rate
- Trades del día
- Operaciones ganadas
- Operaciones perdidas
- Drawdown
- Sharpe
- Última señal
- Operación actual/última

Ejecución:

```bash
AUTO_START_WEB=true bash run.sh
# Dashboard: http://127.0.0.1:9000
```
