CREATE USER trading WITH PASSWORD 'trading';
CREATE DATABASE trading OWNER trading;
\c trading
ALTER SCHEMA public OWNER TO trading;
GRANT ALL ON SCHEMA public TO trading;

CREATE TABLE IF NOT EXISTS orders (
  id BIGSERIAL PRIMARY KEY,
  exchange_order_id TEXT UNIQUE NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  price NUMERIC,
  amount NUMERIC NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fills (
  id BIGSERIAL PRIMARY KEY,
  exchange_order_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  fill_price NUMERIC,
  fill_amount NUMERIC,
  fee NUMERIC,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
