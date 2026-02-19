# reco-trading (producción BTC/USDT Spot 5m)

Sistema cuantitativo modular para Binance Spot con activo único BTC/USDT en timeframe 5m.

## Arquitectura

- `reco_trading/config`: settings y logging.
- `reco_trading/core`: market data, features, modelos (momentum + mean reversion), fusión, riesgo, ejecución, portfolio.
- `reco_trading/infra`: cliente Binance, DB async y persistencia de estado en Redis.
- `reco_trading/research`: backtest profesional, walk-forward y Monte Carlo.
- `reco_trading/monitoring`: health checks y alertas.
- `scripts`: init DB, entrenamiento, backtest, live trading.

## Instalación Debian 13

```bash
./install.sh
cp .env.example .env
# Editar claves reales
./run.sh
```

## Seguridad

- Si `POSTGRES_DSN` apunta a una base inexistente, el sistema intentará crearla automáticamente. Si tu usuario no tiene permisos, define `POSTGRES_ADMIN_DSN` en `.env` con credenciales admin o ejecuta `scripts/init_db.sql`.
- Claves API en `.env` únicamente.
- `.env` ignorado por git.
- Usar API key sin permisos de retiro y con whitelist de IP.
- El sistema separa `BINANCE_TESTNET=false/true` por entorno.

## Solución de autenticación Postgres (usuario `trading`)

Si ves errores tipo `password authentication failed for user "trading"`, ejecuta de nuevo la inicialización:

```bash
sudo -u postgres psql -f scripts/init_db.sql
```

El script ahora **siempre** deja el usuario `trading` con contraseña `trading` (aunque el rol ya exista), para mantener sincronizado el valor usado por defecto en `POSTGRES_DSN`.
