DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'trading') THEN
      CREATE ROLE trading LOGIN;
   END IF;

   ALTER ROLE trading WITH LOGIN PASSWORD 'trading';
END
$$;

SELECT 'CREATE DATABASE trading OWNER trading'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'trading')
\gexec

\c trading

ALTER DATABASE trading OWNER TO trading;
ALTER SCHEMA public OWNER TO trading;
GRANT ALL ON SCHEMA public TO trading;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trading;

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

CREATE TABLE IF NOT EXISTS portfolio_state (
  id BIGSERIAL PRIMARY KEY,
  snapshot JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
