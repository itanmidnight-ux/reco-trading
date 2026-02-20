CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity NUMERIC(24, 12) NOT NULL CHECK (quantity > 0),
    entry_price NUMERIC(24, 12) NOT NULL CHECK (entry_price > 0),
    exit_price NUMERIC(24, 12),
    pnl NUMERIC(24, 12),
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN',
    strategy VARCHAR(64),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    CHECK ((exit_price IS NULL AND closed_at IS NULL) OR (exit_price IS NOT NULL AND closed_at IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    regime VARCHAR(20) NOT NULL,
    probability NUMERIC(10, 9) NOT NULL CHECK (probability >= 0 AND probability <= 1),
    raw_score NUMERIC(24, 12),
    volatility NUMERIC(24, 12) NOT NULL CHECK (volatility >= 0),
    model_breakdown JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio_state (
    id BIGSERIAL PRIMARY KEY,
    equity NUMERIC(24, 12) NOT NULL CHECK (equity >= 0),
    drawdown NUMERIC(10, 9) NOT NULL CHECK (drawdown >= 0 AND drawdown <= 1),
    exposure NUMERIC(10, 9) NOT NULL CHECK (exposure >= 0 AND exposure <= 1),
    var_95 NUMERIC(24, 12),
    leverage NUMERIC(10, 6),
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at);
CREATE INDEX IF NOT EXISTS idx_signals_regime ON signals(regime);
CREATE INDEX IF NOT EXISTS idx_portfolio_state_updated_at ON portfolio_state(updated_at);
