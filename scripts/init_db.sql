\if :{?DB_USER}
\else
\set DB_USER trading
\endif

\if :{?DB_PASSWORD}
\else
\set DB_PASSWORD trading
\endif

\if :{?DB_NAME}
\else
\set DB_NAME reco_trading_prod
\endif

DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = :'DB_USER') THEN
      EXECUTE format('CREATE ROLE %I LOGIN', :'DB_USER');
   END IF;

   EXECUTE format('ALTER ROLE %I WITH LOGIN PASSWORD %L', :'DB_USER', :'DB_PASSWORD');
END
$$;

SELECT format('CREATE DATABASE %I OWNER %I', :'DB_NAME', :'DB_USER')
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'DB_NAME')
\gexec

\connect :DB_NAME

SELECT format('ALTER DATABASE %I OWNER TO %I', :'DB_NAME', :'DB_USER')\gexec
SELECT format('ALTER SCHEMA public OWNER TO %I', :'DB_USER')\gexec
SELECT format('GRANT ALL ON SCHEMA public TO %I', :'DB_USER')\gexec
SELECT format('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO %I', :'DB_USER')\gexec
SELECT format('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO %I', :'DB_USER')\gexec

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
