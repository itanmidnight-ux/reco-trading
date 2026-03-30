# Reco Trading Bot (Binance Spot)

Production-oriented algorithmic trading bot for small-capital spot trading on Binance, built around a state-machine engine, multi-factor voting signals, strict risk controls, and persistent telemetry.

## ⚡️ Características Principales (Versión Avanzada)

### ✨ Nuevas Características Implementadas

#### 🔌 API Server & Web UI
- **FastAPI Server** con endpoints REST
- **WebSocket** para tiempo real
- **CORS** configurado
- **Documentación automática** en `/docs`

#### 🐳 Docker & Multiplataforma
- **Dockerfile** optimizado para producción
- **docker-compose.yml** con PostgreSQL + Redis
- **Ejecución 24/7** en cualquier servidor

#### 🧠 Estrategias Avanzadas
- **30+ indicadores técnicos** (RSI, MACD, EMA, Bollinger Bands, Ichimoku, etc.)
- **Patrones de velas** (Pin Bar, Hammer, Engulfing, Doji)
- **Soporte multi-timeframe**
- **Volume Profile**

#### 🛡️ Sistema de Protecciones (como FreqTrade)
- **Cooldown Period**: Espera después de perder trade
- **Low Profit Pairs**: Bloquea pares con baja ganancia
- **Max Drawdown**: Proteção global contra drawdown
- **Stoploss Guard**: Detiene después de múltiples stoploss

#### 📊 Filtros de Pares (PairList)
- **VolumePairList**: Filtrar por volumen
- **PriceFilter**: Rango de precios
- **SpreadFilter**: Spread máximo
- **VolatilityFilter**: Volatilidad
- **AgeFilter**: Tiempo mínimos cotizando

#### 💰 Gestión de Riesgo Avanzada
- **Capital Profiles** (NANO, MICRO, SMALL, MEDIUM, LARGE, PREMIUM)
- **Partial Take Profit**: 3 niveles de salida
- **Safety Orders (DCA)**: Promediación a la baja
- **Trailing Stop** dinámico

---

## 🚀 Instalación

### 📋 Requisitos Previos

- Python 3.10+ 
- PostgreSQL 14+ (opcional, SQLite por defecto)
- Redis (opcional)
- Docker y Docker Compose (para instalación en contenedor)

### 🐳 Opción 1: Docker (Recomendado)

#### Instalación con Docker Compose

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/reco-trading.git
cd reco-trading

# 2. Configurar variables de entorno
cp .env.example .env
# Edita .env con tus credenciales

# 3. Iniciar servicios
docker-compose up -d

# 4. Ver logs
docker-compose logs -f app

# 5. Detener servicios
docker-compose down
```

#### docker-compose.yml incluye:
- **app**: Reco-Trading bot
- **postgres**: Base de datos PostgreSQL
- **redis**: Cache y colas
- **nginx**: Proxy reverso (opcional)

#### Comandos Docker útiles

```bash
# Iniciar en background
docker-compose up -d

# Ver logs en tiempo real
docker-compose logs -f

# Ver estado de servicios
docker-compose ps

# Reiniciar servicio específico
docker-compose restart app

# Detener y eliminar datos
docker-compose down -v

# Reconstruir imagen
docker-compose build --no-cache
```

#### Puertos expuestos

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| App | 8080 | API REST |
| PostgreSQL | 5432 | Base de datos |
| Redis | 6379 | Cache |

---

### 💻 Opción 2: Instalación Local

#### Linux (Ubuntu/Debian)

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/reco-trading.git
cd reco-trading

# 2. Ejecutar instalador
chmod +x install.sh
./install.sh

# 3. Configurar credenciales
nano .env

# 4. Ejecutar
chmod +x run.sh
./run.sh
```

#### macOS

```bash
# 1. Instalar Homebrew (si no tienes)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Instalar dependencias
brew install python@3.11 postgresql@14 redis

# 3. Clonar y configurar
git clone https://github.com/tu-usuario/reco-trading.git
cd reco-trading

# 4. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 5. Configurar PostgreSQL
brew services start postgresql@14
createdb reco_trading

# 6. Configurar y ejecutar
cp .env.example .env
nano .env
./run.sh
```

#### Windows (WSL2 recomendado)

```bash
# 1. Instalar WSL2
wsl --install -d Ubuntu

# 2. Seguir instrucciones de Linux dentro de WSL
```

---

### ⚙️ Configuración

#### Archivo .env

```env
# ==============================
# RECO TRADING - CONFIGURACIÓN
# ==============================

# Credenciales Binance
BINANCE_API_KEY=tu_api_key_aqui
BINANCE_API_SECRET=tu_api_secret_aqui

# Modo de operación
BINANCE_TESTNET=true          # true = testnet, false = mainnet
ENVIRONMENT=testnet           # testnet o production
RUNTIME_PROFILE=paper         # paper, nano, micro, small, medium, large

# Base de datos
# SQLite (por defecto, recomendado para principiantes)
DATABASE_URL=sqlite:///reco_trading.db

# PostgreSQL (opcional, para producción)
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_USER=trading
# POSTGRES_PASSWORD=tu_password
# POSTGRES_DB=reco_trading_prod

# Redis (opcional)
REDIS_URL=redis://localhost:6379/0

# API Server
API_ENABLED=true
API_HOST=0.0.0.0
API_PORT=8080

# Telegram (opcional)
# TELEGRAM_ENABLED=false
# TELEGRAM_TOKEN=tu_token
# TELEGRAM_CHAT_ID=tu_chat_id
```

#### Obtener credenciales de Binance

1. Ve a [Binance Testnet](https://testnet.binance.vision/)
2. Inicia sesión
3. Ve a API Management
4. Crea una nueva API Key
5. Copia la API Key y Secret al archivo .env

---

### 🎯 Primeros Pasos

#### 1. Verificar instalación

```bash
# Verificar que el bot responde
python -m reco_trading --version

# Ver help
python -m reco_trading --help
```

#### 2. Modo Dry-Run (Recomendado)

```bash
# Iniciar en modo simulación
./run.sh --dry-run

# o con Docker
docker-compose run --rm app trade start --dry-run
```

#### 3. Comandos básicos

```bash
# Ver estado del bot
python -m reco_trading trade status

# Ver estrategias disponibles
python -m reco_trading list-strategies

# Listar exchanges
python -m reco_trading list-exchanges

# Iniciar backtesting
python -m reco_trading backtesting start --strategy DefaultStrategy --timeframe 5m
```

---

### 🔧 Configuración Avanzada

#### Perfiles de Capital

| Perfil | Capital | Riesgo por Trade |
|--------|---------|------------------|
| NANO | $3-25 | 4% |
| MICRO | $25-100 | 3% |
| SMALL | $100-500 | 2% |
| MEDIUM | $500-2000 | 1.5% |
| LARGE | $2000-10000 | 1% |
| PREMIUM | $10000+ | 0.5% |

#### Configurar PostgreSQL

```bash
# Crear base de datos manualmente
psql -U postgres -c "CREATE DATABASE reco_trading_prod;"
psql -U postgres -c "CREATE USER trading WITH PASSWORD 'tu_password';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE reco_trading_prod TO trading;"
```

---

### 🆘 Solución de Problemas

#### Error de conexión a PostgreSQL

```bash
# Verificar que PostgreSQL está corriendo
sudo systemctl status postgresql

# Iniciar PostgreSQL
sudo systemctl start postgresql
```

#### Error de conexión a Redis

```bash
# Verificar Redis
redis-cli ping

# Iniciar Redis
redis-server
```

#### Error de credenciales de Binance

```Asegúrate de que:
1. La API Key tiene permisos de trading
2. La IP está whitelisted (o désactiva-restricción de IP)
3. Estás usando testnet para pruebas

#### Ver logs detallados

```bash
# Docker
docker-compose logs -f app

# Local
python -m reco_trading trade start -v
```

---

## 📊 API Endpoints

| Método | Endpoint | Descripción |
|--------|---------|-------------|
| GET | `/` | Estado del API |
| GET | `/health` | Health check |
| GET | `/api/v1/status` | Estado del bot |
| GET | `/api/v1/balance` | Balance de cuenta |
| GET | `/api/v1/trades` | Historial de trades |
| GET | `/api/v1/open-trades` | Trades abiertos |
| GET | `/api/v1/stats` | Estadísticas |
| GET | `/api/v1/config` | Configuración |
| WS | `/ws` | WebSocket en tiempo real |

---

## ⚙️ Perfiles de Capital

| Perfil | Capital | Riesgo/Trade | Trades/Día |
|--------|---------|--------------|------------|
| NANO | $3-25 | 4% | 2 |
| MICRO | $25-50 | 3% | 2 |
| SMALL | $50-100 | 2.5% | 3 |
| MEDIUM | $100-500 | 2% | 4 |
| LARGE | $500-1000 | 1.5% | 5 |
| PREMIUM | $1000+ | 1.2% | 6 |

---

## 🎯 Filtros por Símbolo (Adaptativos)

| Símbolo | ADX ≥ | RSI Buy | RSI Sell | Volumen |
|---------|--------|---------|----------|---------|
| BTC/USDT | 12 | 48 | 52 | 0.90 |
| ETH/USDT | 12 | 48 | 52 | 0.90 |
| SOL/USDT | 12 | 48 | 52 | 0.85 |
| BNB/USDT | 12 | 48 | 52 | 0.85 |
| XRP/USDT | 10 | 45 | 55 | 0.80 |

---

## 🛡️ Protecciones Activas

1. **Cooldown Period**: 15 minutos entre trades
2. **Daily Loss Limit**: 3% del balance
3. **Max Trades/Day**: Según perfil de capital
4. **Stoploss Guard**: Bloquea después de 3 stoploss consecutivos

---

## 📈 Indicadores Técnicos Disponibles

### Tendencia
- EMA9, EMA20, EMA50, EMA100, EMA200
- Ichimoku Cloud
- KST (Know Sure Thing)

### Momentum
- RSI (14, 28)
- Stochastic
- Williams %R
- Ultimate Oscillator
- ROC (Rate of Change)

### Volatilidad
- ATR, ATR%
- Bollinger Bands (width, position)
- Donchian Channels

### Volumen
- VWAP
- OBV
- ADI
- Chaikin Money Flow
- Volume MA

---

## 🏗️ Arquitectura

```
reco_trading/
├── api/
│   └── server.py           # FastAPI Server
├── core/
│   ├── bot_engine.py       # Motor principal
│   └── state_machine.py    # Estados del bot
├── exchange/
│   ├── binance_client.py   # Cliente Binance
│   └── order_manager.py   # Órdenes
├── strategy/
│   ├── indicators.py      # 30+ indicadores
│   ├── signal_engine.py    # Generación de señales
│   └── confidence_model.py # Modelo de confianza
├── risk/
│   ├── capital_profile.py # Perfiles de capital
│   ├── position_manager.py # Gestión de posiciones
│   └── adaptive_sizer.py  # Tamaño de posición
├── plugins/
│   ├── pairlist.py        # Filtros de pares
│   └── protections.py     # Protecciones
├── data/
│   └── market_stream.py   # Streaming de datos
├── database/
│   ├── models.py          # Modelos SQLAlchemy
│   └── repository.py      # Persistencia
├── ui/
│   └── tabs/             # Interfaz PyQt
└── config/
    └── settings.py        # Configuración
```

---

## 📋 Modos de Ejecución

1. **Testnet**: Simulación con dinero fake
2. **Dry-Run**: Simulación en tiempo real
3. **Live**: Trading real (configurar `BINANCE_TESTNET=false`)

---

## 🔧 Configuración Avanzada

### Docker Compose
```yaml
services:
  app:
    build: .
    ports:
      - "8080:8080"
    env_file: .env
  postgres:
    image: timescale/timescaledb:latest-pg15
  redis:
    image: redis:7-alpine
```

---

## 📝 Changelog

### v2.0.0 (Actual)
- ✅ API Server con FastAPI
- ✅ WebSocket support
- ✅ Docker multiplatform
- ✅ 30+ indicadores técnicos
- ✅ PairList filters (Volume, Price, Spread, Volatility)
- ✅ Protecciones (Cooldown, Max Drawdown, Stoploss Guard)
- ✅ Perfiles de capital optimizados
- ✅ Partial Take Profit
- ✅ Safety Orders (DCA)
- ✅ Filtros adaptativos por símbolo
- ✅ CLI Commands (FreqTrade-style)
- ✅ Config Schema con Pydantic
- ✅ Sistema de validación de configuración
- ✅ Backtesting commands
- ✅ Estratégias commands
- ✅ Worker para lifecycle management
- ✅ Enums para estados del bot
- ✅ DataProvider para estrategias
- ✅ Exchange wrapper (CCXT)
- ✅ IStrategy base class
- ✅ Strategy loader dinámico
- ✅ DefaultStrategy template
- ✅ Backtesting engine
- ✅ Backtesting reports (JSON, texto)
- ✅ CLI backtesting commands
- ✅ FreqAI (Machine Learning)
- ✅ Data Kitchen para ML
- ✅ Modelos: LightGBM, XGBoost, RandomForest
- ✅ Feature Engineering automático
- ✅ Telegram Bot
- ✅ API Server (REST)
- ✅ WebSocket
- ✅ Notifications System
- ✅ RPC Manager

---

## 🖥️ Línea de Comandos (CLI)

Reco-Trading incluye una CLI completa similar a FreqTrade:

```bash
# Iniciar el bot
reco-trade trade start --config config.json

# Ver estado
reco-trade trade status

# Detener el bot
reco-trade trade stop

# Ejecutar backtesting
reco-trade backtesting start --strategy MyStrategy --timeframe 5m

# Crear config por defecto
reco-trade init-config

# Listar estrategias disponibles
reco-trade list-strategies

# Listar exchanges soportados
reco-trade list-exchanges
```

### Comandos Disponibles

| Comando | Descripción |
|---------|-------------|
| `trade start` | Iniciar el bot de trading |
| `trade stop` | Detener el bot |
| `trade status` | Ver estado del bot |
| `trade reload-config` | Recargar configuración |
| `backtesting start` | Ejecutar backtesting |
| `backtesting show` | Mostrar resultados |
| `init-config` | Crear config por defecto |
| `list-strategies` | Listar estrategias |
| `list-exchanges` | Listar exchanges |

---

## 📈 Desarrollo de Estrategias

Reco-Trading utiliza un sistema de estrategias basado en IStrategy:

```python
from reco_trading.strategy import IStrategy

class MiEstrategia(IStrategy):
    # Configuración
    timeframe = "5m"
    stoploss = -0.10
    minimal_roi = {"0": 0.10}
    
    def populate_indicators(self, dataframe, metadata):
        # Añadir indicadores
        dataframe["rsi"] = self._rsi(dataframe["close"])
        return dataframe
    
    def populate_entry_trend(self, dataframe, metadata):
        # Señales de entrada
        dataframe["enter_long"] = 0
        dataframe.loc[dataframe["rsi"] < 30, "enter_long"] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe, metadata):
        # Señales de salida
        dataframe["exit_long"] = 0
        dataframe.loc[dataframe["rsi"] > 70, "exit_long"] = 1
        return dataframe
```

### Atributos de Estrategia

| Atributo | Descripción |
|----------|-------------|
| `timeframe` | Timeframe (1m, 5m, 15m, 1h, etc.) |
| `stoploss` | Stop loss como ratio negativo |
| `minimal_roi` | Tabla de ROI mínimo |
| `trailing_stop` | Habilitar trailing stop |
| `max_open_trades` | Máx. trades abiertos |

### Métodos Required

- `populate_indicators()` - Añadir indicadores técnicos
- `populate_entry_trend()` - Señales de entrada
- `populate_exit_trend()` - Señales de salida

---

## 🔬 Backtesting

Reco-Trading incluye un motor de backtesting para probar estrategias:

```bash
# Ejecutar backtesting
reco-trade backtesting start --strategy DefaultStrategy --timeframe 5m

# Mostrar resultados guardados
reco-trade backtesting show results.json

# Listar resultados
reco-trade backtesting list
```

### Métricas de Backtesting

| Métrica | Descripción |
|---------|-------------|
| Total Trades | Número total de trades ejecutados |
| Winning Trades | Trades con profit positivo |
| Losing Trades | Trades con profit negativo |
| Win Rate | Porcentaje de trades ganadores |
| Total Profit | Ganancia total en stake currency |
| Max Drawdown | Drawdown máximo durante la simulación |
| Avg Duration | Duración promedio de trades |

### Reportes

Los resultados se pueden exportar en JSON:

```python
from reco_trading.backtesting.reports import store_backtest_results

store_backtest_results(results, "backtest_results/my_strategy.json")
```

---

## 🤖 FreqAI - Machine Learning

Reco-Trading incluye FreqAI para predicción con Machine Learning:

```python
# Configuración en config.json
{
    "freqai": {
        "enabled": true,
        "model": "LightGBMClassifier",
        "train_period_days": 14,
        "labeling": {
            "type": "binary",
            "threshold": 0.02
        }
    }
}
```

### Modelos Soportados

| Modelo | Descripción |
|--------|-------------|
| LightGBMClassifier | Clasificador basado en árboles |
| XGBoostClassifier | Clasificador XGBoost |
| RandomForest | Random Forest classifier |

### Feature Engineering

El sistema genera automáticamente:
- Medias móviles (SMA, EMA)
- RSI y MACD
- Bollinger Bands
- Features de volumen
- Features desplazados (lagged)

### Uso

```python
from reco_trading.freqai import FreqAI

freqai = FreqAI(config)
await freqai.start()
predictions = await freqai.predict(dataframe)
```

---

## 📱 Telegram Bot

Reco-Trading incluye un bot de Telegram para control remoto:

```json
{
    "telegram": {
        "enabled": true,
        "token": "YOUR_BOT_TOKEN",
        "chat_id": "YOUR_CHAT_ID"
    }
}
```

### Comandos Disponibles

| Comando | Descripción |
|--------|-------------|
| /start | Iniciar bot |
| /stop | Detener bot |
| /status | Ver estado |
| /profit | Ver ganancias |
| /balance | Ver balance |
| /trades | Ver trades recientes |
| /reload | Recargar config |
| /help | Ver ayuda |

---

## 🌐 API Server

API REST con FastAPI:

```bash
# Endpoints principales
GET /api/v1/status     # Estado del bot
GET /api/v1/balance    # Balance de cuenta
GET /api/v1/trades     # Historial de trades
GET /api/v1/open-trades # Trades abiertos
GET /api/v1/stats      # Estadísticas
GET /api/v1/config     # Configuración
GET /api/v1/pairs     # Lista de pares
GET /ws               # WebSocket para datos en tiempo real
```

### Documentación

Accede a `/docs` para ver la documentación Swagger.

---

## 🔔 Notificaciones

Sistema de notificaciones integrado:

- Telegram
- Webhooks
- Notificaciones de entry/exit
- Alertas de protecciones
- Errores y warnings

```python
from reco_trading.rpc import NotificationManager

notifications = NotificationManager(config)
await notifications.notify_entry(trade)
await notifications.notify_exit(trade)
```

---

## ⚠️ Disclaimer

Este bot es para fines educativos. Úsalo bajo tu propio riesgo. Siempre haz pruebas en testnet antes de usar dinero real.

---

## 📧 Soporte

- GitHub: [reco-trading](https://github.com)
- Issues: Reporta errores y sugerencias
