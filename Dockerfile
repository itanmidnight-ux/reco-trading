# Reco-Trading Dockerfile
# Multi-stage build for production with Web Dashboard and Cloudflare Tunnel
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -t /packages

FROM python:3.11-slim

LABEL maintainer="Reco-Trading Team"
LABEL description="Reco-Trading - Advanced Cryptocurrency Trading Bot"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DASHBOARD_TYPE=web

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 -s /bin/bash appuser

# Install cloudflared tunnel (try from official source, fallback to direct download)
RUN curl -sSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared \
    && chmod +x /usr/local/bin/cloudflared \
    || (wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared \
    && chmod +x /usr/local/bin/cloudflared)

COPY --from=builder /packages /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

# Copy .env from host to container (will be mounted at runtime)
# But also ensure we have a default if not provided
RUN mkdir -p /app/data /app/logs && \
    chown -R appuser:appuser /app

USER appuser

# Expose web dashboard port and cloudflared metrics
EXPOSE 9000 8080 9090

# Healthcheck for web dashboard
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:9000/api/health || exit 1

# Start both the bot and cloudflared tunnel
# The bot will run in web dashboard mode by default (DASHBOARD_TYPE=web)
# cloudflared runs in background, outputting URL to stdout
CMD ["sh", "-c", "\
    echo 'Iniciando Cloudflare Tunnel...'; \
    cloudflared tunnel --url http://localhost:9000 --metrics 0.0.0.0:9090 2>&1 & \
    cloudflared_pid=$!; \
    echo 'Iniciando Reco-Trading Bot...'; \
    python -m reco_trading.main; \
    kill $cloudflared_pid 2>/dev/null || true"]