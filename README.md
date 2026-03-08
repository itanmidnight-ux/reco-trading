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
