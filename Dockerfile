# Reco-Trading Dockerfile v4.0
# Multi-stage build with auto system detection and all AI/ML features enabled
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -t /packages

FROM python:3.11-slim

LABEL maintainer="Reco-Trading Team"
LABEL description="Reco-Trading - Advanced AI Cryptocurrency Trading Bot"

# Environment variables for Python optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DASHBOARD_TYPE=web \
    RESOURCE_PROFILE=auto \
    WORKERS=auto \
    ML_WORKERS=auto \
    ENABLE_AUTO_IMPROVER=true \
    ENABLE_ML_ENGINE=true \
    ENABLE_CONTINUAL_LEARNING=true \
    ENABLE_META_LEARNING=true \
    ENABLE_DRIFT_DETECTION=true \
    ENABLE_ONCHAIN_ANALYSIS=true \
    ENABLE_EVOLUTION=true \
    ENABLE_TFT=true \
    ENABLE_NBEATS=true \
    ENABLE_ADVANCED_META_LEARNING=true \
    ENABLE_REINFORCEMENT_LEARNING=true \
    DATABASE_URL=sqlite+aiosqlite:////app/data/reco_trading.db

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 -s /bin/bash appuser

# Install TA-Lib for technical analysis
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tar.gz \
    && tar -xzf /tmp/ta-lib.tar.gz -C /tmp \
    && cd /tmp/ta-lib \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && cd / \
    && rm -rf /tmp/ta-lib* \
    || echo "TA-Lib installation failed, using fallback"

# Copy Python packages from builder
COPY --from=builder /packages /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/cache && \
    chown -R appuser:appuser /app

USER appuser

# Expose ports
EXPOSE 9000 8080 9090

# Healthcheck for web dashboard
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9000/api/health 2>/dev/null || exit 1

# Start script that initializes all features and runs cloudflared tunnel
COPY --chmod=755 <<'EOF' /app/start.sh
#!/bin/bash
set -e

echo "=================================================="
echo "  Reco-Trading Bot - Starting..."
echo "=================================================="
echo ""

# Detect system resources
CPU_CORES=$(nproc 2>/dev/null || echo "2")
TOTAL_MEM=$(free -m 2>/dev/null | awk '/Mem:/ {print $2}' || echo "2048")
RAM_GB=$((TOTAL_MEM / 1024))

echo "System Resources:"
echo "  CPU Cores: $CPU_CORES"
echo "  RAM: ${RAM_GB}GB"
echo ""

# Determine resource profile
if [[ $RAM_GB -ge 16 ]] && [[ $CPU_CORES -ge 4 ]]; then
    export RESOURCE_PROFILE="${RESOURCE_PROFILE:-high}"
    export WORKERS="${WORKERS:-4}"
    export ML_WORKERS="${ML_WORKERS:-4}"
    echo "Resource Profile: HIGH"
elif [[ $RAM_GB -ge 8 ]] && [[ $CPU_CORES -ge 2 ]]; then
    export RESOURCE_PROFILE="${RESOURCE_PROFILE:-medium}"
    export WORKERS="${WORKERS:-2}"
    export ML_WORKERS="${ML_WORKERS:-2}"
    echo "Resource Profile: MEDIUM"
else
    export RESOURCE_PROFILE="${RESOURCE_PROFILE:-low}"
    export WORKERS="${WORKERS:-1}"
    export ML_WORKERS="${WORKERS:-1}"
    echo "Resource Profile: LOW (optimized)"
fi

# Enable all AI/ML features
export ENABLE_AUTO_IMPROVER="${ENABLE_AUTO_IMPROVER:-true}"
export ENABLE_ML_ENGINE="${ENABLE_ML_ENGINE:-true}"
export ENABLE_CONTINUAL_LEARNING="${ENABLE_CONTINUAL_LEARNING:-true}"
export ENABLE_META_LEARNING="${ENABLE_META_LEARNING:-true}"
export ENABLE_DRIFT_DETECTION="${ENABLE_DRIFT_DETECTION:-true}"
export ENABLE_ONCHAIN_ANALYSIS="${ENABLE_ONCHAIN_ANALYSIS:-true}"
export ENABLE_EVOLUTION="${ENABLE_EVOLUTION:-true}"
export ENABLE_TFT="${ENABLE_TFT:-true}"
export ENABLE_NBEATS="${ENABLE_NBEATS:-true}"
export ENABLE_ADVANCED_META_LEARNING="${ENABLE_ADVANCED_META_LEARNING:-true}"
export ENABLE_REINFORCEMENT_LEARNING="${ENABLE_REINFORCEMENT_LEARNING:-true}"

echo ""
echo "AI/ML Features Enabled:"
[[ "$ENABLE_AUTO_IMPROVER" == "true" ]] && echo "  ✓ Auto-Improver"
[[ "$ENABLE_ML_ENGINE" == "true" ]] && echo "  ✓ ML Engine"
[[ "$ENABLE_CONTINUAL_LEARNING" == "true" ]] && echo "  ✓ Continual Learning"
[[ "$ENABLE_META_LEARNING" == "true" ]] && echo "  ✓ Meta-Learning"
[[ "$ENABLE_DRIFT_DETECTION" == "true" ]] && echo "  ✓ Drift Detection"
[[ "$ENABLE_ONCHAIN_ANALYSIS" == "true" ]] && echo "  ✓ On-Chain Analysis"
[[ "$ENABLE_EVOLUTION" == "true" ]] && echo "  ✓ Evolution System"
[[ "$ENABLE_TFT" == "true" ]] && echo "  ✓ Temporal Fusion Transformer (TFT)"
[[ "$ENABLE_NBEATS" == "true" ]] && echo "  ✓ N-BEATS Model"
[[ "$ENABLE_ADVANCED_META_LEARNING" == "true" ]] && echo "  ✓ Advanced Meta-Learning"
[[ "$ENABLE_REINFORCEMENT_LEARNING" == "true" ]] && echo "  ✓ Reinforcement Learning (PPO/TD3)"
echo ""

# Start cloudflared tunnel in background
echo "Starting Cloudflare Tunnel..."
cloudflared tunnel --url http://localhost:9000 --metrics 0.0.0.0:9090 2>&1 &
CLOUDFLARED_PID=$!

# Wait for cloudflared to be ready
sleep 5

# Start the bot
echo "Starting Reco-Trading Bot..."
echo "=================================================="
echo ""

exec /app/.venv/bin/python -m reco_trading.main
EOF

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]