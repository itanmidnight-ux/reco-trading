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

CREATE TABLE IF NOT EXISTS system_config_versions (
    id BIGSERIAL PRIMARY KEY,
    version VARCHAR(64) NOT NULL UNIQUE,
    config_hash VARCHAR(128) NOT NULL,
    signature TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'failed', 'rolled_back')),
    reason TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    activated_at TIMESTAMP,
    failed_at TIMESTAMP,
    rolled_back_at TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS system_config_snapshots (
    id BIGSERIAL PRIMARY KEY,
    version_id BIGINT NOT NULL REFERENCES system_config_versions(id) ON DELETE CASCADE,
    snapshot_hash VARCHAR(128) NOT NULL,
    snapshot_signature TEXT NOT NULL,
    snapshot_payload JSONB NOT NULL,
    reason TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS system_deployments (
    id BIGSERIAL PRIMARY KEY,
    version_id BIGINT NOT NULL REFERENCES system_config_versions(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'active', 'failed', 'rolled_back')),
    reason TEXT,
    deployed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS system_rollbacks (
    id BIGSERIAL PRIMARY KEY,
    deployment_id BIGINT REFERENCES system_deployments(id) ON DELETE SET NULL,
    from_version_id BIGINT NOT NULL REFERENCES system_config_versions(id) ON DELETE CASCADE,
    to_version_id BIGINT NOT NULL REFERENCES system_config_versions(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'active', 'failed', 'rolled_back')),
    reason TEXT NOT NULL,
    triggered_by VARCHAR(32) NOT NULL DEFAULT 'automatic',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    id BIGSERIAL PRIMARY KEY,
    exchange_order_id VARCHAR(128) NOT NULL UNIQUE,
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(10) NOT NULL,
    price NUMERIC(24, 12),
    amount NUMERIC(24, 12) NOT NULL,
    status VARCHAR(32) NOT NULL,
    decision_id VARCHAR(64),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fills (
    id BIGSERIAL PRIMARY KEY,
    exchange_order_id VARCHAR(128) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(10) NOT NULL,
    fill_price NUMERIC(24, 12),
    fill_amount NUMERIC(24, 12),
    fee NUMERIC(24, 12),
    order_id BIGINT,
    decision_id VARCHAR(64),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS order_executions (
    id BIGSERIAL PRIMARY KEY,
    ts BIGINT NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(10) NOT NULL,
    qty NUMERIC(24, 12) NOT NULL,
    price NUMERIC(24, 12) NOT NULL,
    status VARCHAR(32) NOT NULL,
    exchange_order_id VARCHAR(128),
    pnl NUMERIC(24, 12) NOT NULL DEFAULT 0,
    order_id BIGINT,
    decision_id VARCHAR(64),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at);
CREATE INDEX IF NOT EXISTS idx_signals_regime ON signals(regime);
CREATE INDEX IF NOT EXISTS idx_portfolio_state_updated_at ON portfolio_state(updated_at);
CREATE INDEX IF NOT EXISTS idx_system_config_versions_status ON system_config_versions(status);
CREATE INDEX IF NOT EXISTS idx_system_config_versions_created_at ON system_config_versions(created_at);
CREATE INDEX IF NOT EXISTS idx_system_snapshots_version_id ON system_config_snapshots(version_id);
CREATE INDEX IF NOT EXISTS idx_system_deployments_version_id ON system_deployments(version_id);
CREATE INDEX IF NOT EXISTS idx_system_rollbacks_from_version_id ON system_rollbacks(from_version_id);

CREATE TABLE IF NOT EXISTS execution_idempotency_ledger (
    id BIGSERIAL PRIMARY KEY,
    client_order_id VARCHAR(64) NOT NULL UNIQUE,
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(10) NOT NULL,
    qty NUMERIC(24, 12) NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('PENDING_SUBMIT','SUBMITTED','PARTIALLY_FILLED','FILLED','CANCELLED','FAILED','SUBMISSION_UNCERTAIN')),
    exchange_order_id VARCHAR(64) NOT NULL DEFAULT '',
    decision_id VARCHAR(64),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS capital_reservations (
    id BIGSERIAL PRIMARY KEY,
    reservation_id VARCHAR(128) NOT NULL UNIQUE,
    client_order_id VARCHAR(64) NOT NULL UNIQUE,
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(10) NOT NULL,
    reserved_amount NUMERIC(24, 12) NOT NULL,
    used_amount NUMERIC(24, 12) NOT NULL DEFAULT 0,
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE IF EXISTS capital_reservations
    ADD COLUMN IF NOT EXISTS client_order_id VARCHAR(64);

UPDATE capital_reservations
SET client_order_id = COALESCE(NULLIF(client_order_id, ''), reservation_id)
WHERE client_order_id IS NULL OR client_order_id = '';

CREATE UNIQUE INDEX IF NOT EXISTS ux_capital_reservations_client_order_id
    ON capital_reservations(client_order_id);

CREATE TABLE IF NOT EXISTS decision_audit (
    id BIGSERIAL PRIMARY KEY,
    decision_id VARCHAR(64) NOT NULL UNIQUE,
    snapshot JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE IF EXISTS orders ADD COLUMN IF NOT EXISTS decision_id VARCHAR(64);
ALTER TABLE IF EXISTS fills ADD COLUMN IF NOT EXISTS decision_id VARCHAR(64);
ALTER TABLE IF EXISTS order_executions ADD COLUMN IF NOT EXISTS decision_id VARCHAR(64);

DO
$$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'orders'
          AND column_name = 'decision_id'
    ) THEN
        ALTER TABLE orders ADD COLUMN decision_id VARCHAR(64);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_orders_decision_id') THEN
        ALTER TABLE orders
            ADD CONSTRAINT fk_orders_decision_id
            FOREIGN KEY (decision_id) REFERENCES decision_audit(decision_id) ON DELETE SET NULL;
    END IF;
END
$$;

ALTER TABLE IF EXISTS fills
    ADD COLUMN IF NOT EXISTS order_id BIGINT;

DO
$$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'fills'
          AND column_name = 'order_id'
    ) THEN
        ALTER TABLE fills ADD COLUMN order_id BIGINT;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_fills_order_id') THEN
        ALTER TABLE fills
            ADD CONSTRAINT fk_fills_order_id
            FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE SET NULL;
    END IF;
END
$$;

ALTER TABLE IF EXISTS order_executions
    ADD COLUMN IF NOT EXISTS order_id BIGINT;

DO
$$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'order_executions'
          AND column_name = 'order_id'
    ) THEN
        ALTER TABLE order_executions ADD COLUMN order_id BIGINT;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_order_executions_order_id') THEN
        ALTER TABLE order_executions
            ADD CONSTRAINT fk_order_executions_order_id
            FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE SET NULL;
    END IF;
END
$$;

ALTER TABLE IF EXISTS order_executions ADD COLUMN IF NOT EXISTS exchange_order_id VARCHAR(64);
