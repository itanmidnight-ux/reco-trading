# reco-trading: Minimal Binance Retail Bot

This refactor replaces the institutional multi-module stack with a minimal, safety-first bot for very small accounts.

## New architecture

```text
bot/
  config.py
  main.py
  exchange/
    binance_client.py
    websocket_manager.py
  core/
    market_data.py
    strategy.py
    risk_manager.py
    execution_engine.py
    portfolio.py
  services/
    order_manager.py
    state_manager.py
  utils/
    logger.py
    helpers.py
```

## Strategy (single low-frequency directional model)

- Momentum + SMA trend confirmation (fast/slow SMA)
- Volatility filter (minimum and maximum volatility regime)
- RSI filter to avoid overstretched entries
- Trade target: low-frequency operation (3–10 trades/week target with 15m bars + strict risk gates)

## Risk constraints

- `max_risk_per_trade = 1%`
- `max_trades_per_day = 3`
- `daily_loss_limit = 3%`
- `max_open_positions = 1`
- Trading stops for the day when daily loss limit is reached.

## Execution safety

Before any order, the bot validates:

- LOT_SIZE-adjusted quantity
- MIN_NOTIONAL
- PRICE reference
- available balance
- expected edge > estimated fees + spread + extra safety buffer

## Binance safety controls

- `recvWindow` on signed order calls
- timestamp drift sync (`fetch_time` + local offset)
- retry with exponential backoff for API operations
- websocket manager with auto-reconnect
- order reconciliation through `fetch_my_trades`

## Run

```bash
pip install -r requirements.txt
export BINANCE_API_KEY="..."
export BINANCE_API_SECRET="..."
export BINANCE_TESTNET=true
python -m bot.main
```

Or run the root entrypoint:

```bash
python main.py
```
