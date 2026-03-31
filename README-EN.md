# Reco-Trading Bot - Complete Documentation (English)

> **Version**: 2.0.0  
> **Last Updated**: March 31, 2026

Reco-Trading is an enterprise-grade autonomous cryptocurrency trading bot with advanced AI/ML capabilities. It integrates Bayesian optimization, genetic algorithms, reinforcement learning, Hidden Markov Models for market regime detection, and adaptive filters - making it the most sophisticated open-source trading bot available.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Bot](#running-the-bot)
5. [Web Dashboard](#web-dashboard)
6. [Backtesting](#backtesting)
7. [AI/ML System](#aiml-system)
8. [Multi-Pair Management](#multi-pair-management)
9. [Autonomous Brain](#autonomous-brain)
10. [Security](#security)
11. [Comparison](#comparison)
12. [Support](#support)

---

## Features

### Core Trading Features
- **Multi-Pair Trading**: 104 trading pairs across 10 priority tiers
- **Smart Pair Switching**: Advanced algorithm with circuit breaker protection
- **Tier-Based Scanning**: Optimized scan frequencies (10s - 60s) based on pair tier
- **Real-Time Market Analysis**: HMM-based regime detection (Bull/Bear/Sideways/HighVol/LowVol)

### AI/ML Capabilities (Industry-Leading)
- **Bayesian Optimization**: Gaussian Process-based hyperparameter tuning
- **Genetic Algorithm**: Multi-objective evolutionary strategy optimization
- **Q-Learning Agent**: Reinforcement learning for decision making
- **Policy Gradient (REINFORCE)**: Policy-based RL for trading
- **Ensemble Predictor**: Combines 4+ models for robust signals
- **Market Regime Detection**: Hidden Markov Models with 5 states

### Risk Management
- **Circuit Breaker**: Automatic pause after 5 consecutive errors (300s)
- **Panic Threshold**: Auto-switch pair when opportunity score < 0.20
- **Dynamic Position Sizing**: Adjusts based on market conditions & confidence
- **Adaptive Filters**: Kalman and KAMA filters for noise reduction
- **Consecutive Loss Protection**: Switch after 2 losses

### Interfaces
- **Web Dashboard**: Flask-based UI on port 9000
- **PySide6 App**: Native desktop application
- **REST API**: Full programmatic access
- **Docker Support**: Production-ready containers

---

## Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 15+ (optional, for persistence)
- Redis 7+ (optional, for caching)
- Docker & Docker Compose (optional)

### Quick Install - Windows

```batch
# Clone or download the repository
cd reco-trading

# Run the launcher (automatically creates venv and installs deps)
reco.bat dry
```

### Quick Install - Linux/macOS

```bash
# Clone or download the repository
cd reco-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run in dry-run mode (recommended first)
python -m reco_trading.main
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -f Dockerfile.backtest -t reco-trading .
docker run -it reco-trading python -m reco_trading.main
```

---

## Configuration

### Environment Variables (.env)

Create a `.env` file in the root directory:

```env
# Exchange API Keys (REQUIRED for live trading)
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET=your_secret_here

# Database (optional)
POSTGRES_PASSWORD=your_secure_password

# Bot Settings
DRY_RUN=true
LOG_LEVEL=INFO
LOG_FILE=logs/reco_trading.log

# Dashboard
DASHBOARD_PORT=9000
DASHBOARD_HOST=0.0.0.0
```

### YAML Configuration (config/settings.yaml)

Create `config/settings.yaml`:

```yaml
# Trading Configuration
trading:
  symbol: BTC/USDT          # Primary trading pair
  timeframe: 5m             # Candle timeframe
  dry_run: true             # Test mode (no real trades)
  initial_capital: 1000.0  # Starting capital
  max_position_size: 0.1   # Max 10% of capital per trade

# Autonomous Brain Configuration
autonomous:
  enabled: true
  decision_interval: 30     # Decision loop every 30 seconds
  optimization_interval: 4    # Parameter optimization every 4 hours
  emergency_pause_after_losses: 5
  auto_resume_after_wins: 3
  max_daily_trades: 20

# Multi-Pair Management
multipair:
  enabled: true
  pairs_count: 104         # Total pairs to monitor
  scan_interval: 15         # Base scan interval in seconds
  switch_after_losses: 2    # Switch after 2 consecutive losses
  circuit_breaker_duration: 300  # 5 minutes
  panic_threshold: 0.20      # Switch if opportunity < 0.20
  min_volume_24h: 2000000  # Minimum 24h volume
  max_volatility: 0.12     # Max volatility threshold
  
  # Tier-based scan intervals (seconds)
  tier_intervals:
    1: 10   # Top 10 pairs
    2: 15   # Pairs 11-20
    3: 20   # Pairs 21-32
    4: 30   # Pairs 33-52
    5: 45   # Pairs 53-72
    6: 60   # Remaining pairs

# AI/ML System
ai:
  bayesian:
    enabled: true
    exploration_weight: 0.15
    max_iterations: 50
    
  genetic:
    enabled: true
    population_size: 15
    elite_size: 2
    mutation_rate: 0.15
    crossover_rate: 0.75
    generations: 30
    
  reinforcement_learning:
    enabled: true
    learning_rate: 0.1
    discount_factor: 0.95
    epsilon: 1.0
    epsilon_decay: 0.995
    epsilon_min: 0.01
    
  regime_detection:
    enabled: true
    n_states: 5
    confidence_threshold: 0.7

# Trading Filters
filters:
  rsi_period: 14
  rsi_overbought: 70
  rsi_oversold: 30
  stop_loss: 2.0
  take_profit: 4.0
  min_confidence: 0.70
  ma_short_period: 9
  ma_long_period: 21
```

---

## Running the Bot

### Windows

```batch
# Dry-run mode (RECOMMENDED for first run)
reco.bat dry

# Live trading (with real money)
reco.bat run

# Web dashboard only
reco.bat dashboard

# Backtesting
reco.bat backtest --pairs BTC/USDT ETH/USDT --days 30

# Custom configuration
reco.bat run --config config/production.yaml
```

### Linux/macOS

```bash
# Activate virtual environment
source venv/bin/activate

# Dry-run mode (recommended)
python -m reco_trading.main --dry-run

# Live trading
python -m reco_trading.main

# With custom config
python -m reco_trading.main --config config/production.yaml
```

### Docker

```bash
# Dry-run in container
docker-compose run --rm app python -m reco_trading.main --dry-run

# Live mode
docker-compose run -e DRY_RUN=false app python -m reco_trading.main

# Backtesting
docker-compose run --rm backtest python -m reco_trading.backtest --symbol BTC/USDT --days 30
```

---

## Web Dashboard

The web dashboard provides comprehensive real-time monitoring and control.

### Access

```
http://localhost:9000
```

### Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | Real-time status, price, signals, account info |
| **Trades** | Trade history with details and P&L |
| **Analytics** | Performance charts, win rate, Sharpe ratio |
| **Market** | Market data, regime, volatility |
| **Strategy** | Current strategy parameters |
| **AI/ML** | Bayesian optimizer, GA, regime detector, adaptive filters |
| **Risk** | Risk metrics, drawdown, position limits |
| **Health** | System health, connections, latency |
| **Logs** | Application logs |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/snapshot` | GET | Current bot snapshot |
| `/api/status` | GET | Bot status |
| `/api/health` | GET | Health check |
| `/api/autonomous` | GET | Autonomous brain status |
| `/api/multipair` | GET | Multi-pair manager status |
| `/api/trades` | GET | Recent trades |
| `/api/all_trades` | GET | All trades from database |
| `/api/analytics` | GET | Analytics data |
| `/api/settings` | GET | Current settings |
| `/api/control/<action>` | POST | Control (pause/resume/emergency) |

### Control Actions

```bash
# Pause the bot
curl -X POST http://localhost:9000/api/control/pause

# Resume trading
curl -X POST http://localhost:9000/api/control/resume

# Emergency stop
curl -X POST http://localhost:9000/api/control/emergency
```

---

## Backtesting

### Quick Start

```bash
# Basic backtest
python -m reco_trading.backtest --symbol BTC/USDT --days 30

# Multiple pairs
python -m reco_trading.backtest --pairs BTC/USDT ETH/USDT SOL/USDT --days 90

# With parameter optimization
python -m reco_trading.backtest --symbol BTC/USDT --days 90 --optimize

# Custom initial capital
python -m reco_trading.backtest --symbol BTC/USDT --days 30 --initial-capital 5000
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--symbol` | Trading pair | BTC/USDT |
| `--pairs` | Multiple pairs (comma-separated) | Single symbol |
| `--days` | Historical days to backtest | 30 |
| `--timeframe` | Candle timeframe (1m/5m/15m/1h/4h/1d) | 5m |
| `--initial-capital` | Starting capital | 1000.0 |
| `--optimize` | Run parameter optimization | False |
| `--output` | Output results file | results.json |
| `--commission` | Commission rate (0.001 = 0.1%) | 0.001 |

### Backtest Results

The backtester outputs:

```json
{
  "summary": {
    "total_trades": 150,
    "winning_trades": 95,
    "losing_trades": 55,
    "win_rate": 0.633,
    "total_pnl": 1250.50,
    "total_pnl_percent": 125.05,
    "avg_trade_pnl": 8.34,
    "best_trade": 45.20,
    "worst_trade": -15.30
  },
  "metrics": {
    "sharpe_ratio": 1.85,
    "sortino_ratio": 2.34,
    "max_drawdown": 0.12,
    "calmar_ratio": 1.45,
    "profit_factor": 1.85
  },
  "trades": [...]
}
```

---

## AI/ML System

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE PREDICTOR                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │   BAYESIAN    │  │    GENETIC    │  │     RL       │  │
│  │  OPTIMIZER    │  │   ALGORITHM   │  │    AGENT     │  │
│  │  (Parameter   │  │  (Evolution)  │  │ (Q-Learning) │  │
│  │   Tuning)     │  │               │  │              │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │     HMM       │  │   TECHNICAL   │  │    KALMAN    │  │
│  │    REGIME     │  │  INDICATORS  │  │    FILTER    │  │
│  │  Detection    │  │    (RSI,      │  │  Noise       │  │
│  │               │  │   MACD, etc)   │  │  Reduction   │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Status | Description |
|-----------|--------|-------------|
| **Bayesian Optimizer** | ✅ Active | Gaussian Process-based parameter tuning with Expected Improvement acquisition |
| **Genetic Algorithm** | ✅ Active | Tournament selection, single-point crossover, Gaussian mutation |
| **Q-Learning Agent** | ✅ Active | Reinforcement learning with epsilon-greedy exploration |
| **Policy Gradient** | ✅ Active | REINFORCE algorithm for policy learning |
| **Ensemble Predictor** | ✅ Active | Weighted voting from all models |
| **HMM Regime Detector** | ✅ Active | 5-state Hidden Markov Model (Bull/Bear/Sideways/HighVol/LowVol) |
| **Kalman Filter** | ✅ Active | Optimal state estimation for price |
| **KAMA Filter** | ✅ Active | Kaufman Adaptive Moving Average |

### Auto-Optimization Schedule

| Process | Frequency | Description |
|---------|-----------|-------------|
| **Decision Loop** | Every 30s | Analyze performance, adjust confidence |
| **Parameter Optimization** | Every 4h | Bayesian + Genetic optimization |
| **Trade Recording** | On each trade | Record for learning |
| **Regime Detection** | On new data | Update market regime |
| **Filter Adaptation** | Every 60s | Adjust Kalman/KAMA |

---

## Multi-Pair Management

### Tier System (104 Pairs)

| Tier | Pairs | Scan Interval | Example Pairs |
|------|-------|---------------|----------------|
| 1 | 10 | 10s | BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, DOT, LINK |
| 2 | 10 | 15s | MATIC, ATOM, UNI, XLM, ETC, ALGO, FIL, HBAR, NEAR, APT |
| 3 | 12 | 20s | ARB, OP, SUI, INJ, SEI, TIA, PEPE, WIF, SSV, FTM, JUP, WLD |
| 4-10 | 72 | 30-60s | DeFi, Layer2, Gaming, AI tokens |

### Smart Pair Switching Algorithm

The switching algorithm considers multiple factors:

```
Switch Score = 
  Opportunity Score × 0.40 +
  Momentum Advantage × 0.20 +
  Volume Advantage × 0.15 +
  Volatility Advantage × 0.10 +
  Regime Match Bonus × 0.10 +
  Risk-Adjusted Return × 0.05
```

### Protection Mechanisms

| Mechanism | Value | Description |
|-----------|-------|-------------|
| Panic Threshold | 0.20 | Auto-switch if opportunity < 0.20 |
| Circuit Breaker | 300s | Pause for 5 min after errors |
| Min Hold Time | 300s | Minimum 5 min per pair |
| Max Switches/Hour | 3 | Prevents pair hopping |
| Consecutive Losses | 2 | Trigger switch after 2 losses |

---

## Autonomous Brain

The autonomous brain orchestrates all AI/ML systems and makes independent trading decisions without human intervention.

### Self-Healing Capabilities

- **Auto-Reconnect**: Reconnects on network errors
- **Auto-Restart**: Recovers from crashes gracefully
- **Auto-Adjust**: Modifies parameters based on performance
- **Auto-Backup**: Saves state periodically

### Decision Loop Flow

```
1. Analyze recent performance (win rate, P&L)
   ↓
2. Adjust confidence threshold (if needed)
   ↓
3. Detect market regime (HMM)
   ↓
4. Apply regime-based filters
   ↓
5. Run AI/ML ensemble prediction
   ↓
6. Execute trade decision (if signal)
   ↓
7. Record trade for learning
   ↓
8. Optimize parameters (every 4 hours)
```

---

## Security

### Best Practices

1. **API Keys**: Never commit to version control
2. **Environment**: Use `.env` file or environment variables
3. **Permissions**: Enable IP restrictions on exchanges
4. **Read-Only**: Use read-only keys when possible
5. **Start Dry**: Always test in dry-run mode first

### Recommended Setup

```env
# Use separate keys for trading bot
# Enable only: Spot Trading, Margin Trading (if needed)
# Enable IP restriction to your server IP
# Disable withdrawals
```

---

## Comparison with Other Bots

### Comparison Table v2.0 (2026)

| Feature | Reco-Trading Pro | Freqtrade | Gunbot | 3Commas | Zenbot |
|---------|------------------|-----------|--------|---------|--------|
| **Multi-Pair** | 104 | 100+ | 100+ | 100+ | 50+ |
| **Bayesian Optimization** | ✅ Advanced | ✅ (FreqAI) | ❌ | ⚠️ Basic | ❌ |
| **Genetic Algorithm** | ✅ + Novelty Search | ❌ | ❌ | ❌ | ❌ |
| **Reinforcement Learning** | ✅ Q-Learning + Policy Gradient | ❌ | ❌ | ❌ | ❌ |
| **LLM Agents** | ✅ Analyst + Risk + Executor | ❌ | ❌ | ❌ | ❌ |
| **Meta-Learning (MAML)** | ✅ Fast Adaptation | ❌ | ❌ | ❌ | ❌ |
| **Continual Learning** | ✅ Online + Experience Replay | ❌ | ❌ | ❌ | ❌ |
| **Test-Time Adaptation** | ✅ Drift Detection | ❌ | ❌ | ❌ | ❌ |
| **On-Chain Analysis** | ✅ Whale + Flow + Smart Money | ❌ | ❌ | ❌ | ❌ |
| **HMM Regime Detection** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Kalman Filter** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **KAMA Filter** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Circuit Breaker** | ✅ Advanced | ✅ | ✅ | ✅ | ✅ |
| **Web Dashboard** | ✅ + AI/ML Tab | ✅ | ✅ | ✅ | ⚠️ Basic |
| **Docker Support** | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Open Source** | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Self-Evolving** | ✅ Auto-Optimize | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual | ❌ |
| **Monthly Cost** | FREE | FREE | $30-100 | $30-100 | FREE |

### Advanced Features Comparison (2026)

| AI/ML Capability | Reco-Trading Pro | Binance AI Pro | PionexGPT | MAKORA |
|------------------|------------------|----------------|-----------|--------|
| **Multi-Agent System** | ✅ 3 Agents | ✅ | ⚠️ Signal-based | ✅ |
| **Meta-Learning** | ✅ MAML-style | ❌ | ❌ | ❌ |
| **Continual Learning** | ✅ Online | ❌ | ❌ | ❌ |
| **Drift Detection** | ✅ Performance + Data | ❌ | ❌ | ❌ |
| **On-Chain Signals** | ✅ Whale/Flow | ⚠️ Limited | ❌ | ❌ |
| **Novelty Search** | ✅ Escape Optima | ❌ | ❌ | ❌ |
| **Knowledge Distillation** | ✅ | ❌ | ❌ | ❌ |
| **Autonomous Decision** | ✅ Full Auto | ✅ | ⚠️ Semi | ✅ |

### Unique Features (Reco-Trading Pro)

1. **LLM Agents**: Multi-agent system with Analyst, Risk Manager, and Executor
2. **Meta-Learning**: Fast adaptation to new markets in 5-10 gradient steps
3. **Continual Learning**: Online learning with catastrophic forgetting prevention
4. **Test-Time Adaptation**: Real-time drift detection and model adjustment
5. **On-Chain Intelligence**: Whale tracking, exchange flows, smart money detection
6. **Novelty Search**: Evolution escapes local optima automatically
7. **Zero-Config Operation**: Fully autonomous, no manual intervention needed

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Module import error | Run `pip install -r requirements.txt` |
| Connection refused | Check exchange API status and internet |
| No trades executed | Lower confidence threshold in settings |
| High drawdown | Reduce position size, increase stop loss |
| Memory leak | Restart bot periodically |

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python -m reco_trading.main

# Or in .env
LOG_LEVEL=DEBUG
```

### Log Locations

- Application logs: `logs/reco_trading.log`
- Trade history: Database (PostgreSQL)
- Dashboard logs: `logs/web_dashboard.log`

---

## License

MIT License - See LICENSE file for details.

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/anomalyco/reco-trading/issues
- Documentation: https://reco-trading.readthedocs.io

---

**Version 2.0.0** - Built with ❤️ for the crypto trading community

*This documentation is continuously updated. Last sync: March 31, 2026*
