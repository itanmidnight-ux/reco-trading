# Sistema Automatizado de Trading Cuantitativo Avanzado

Implementación profesional, modular, asincrónica y orientada a eventos para trading algorítmico con Binance, preparada para operación 24/7 y despliegue dockerizado.

## Capacidades principales

- Ingesta en tiempo real por WebSocket (`kline`, `trade`, `depth`) y REST selectivo.
- Construcción de histórico OHLCV para features y señales.
- Feature engineering técnico, estadístico y microestructural con confirmación estadística (t-stat, pseudo-p-value, hurst, estacionariedad).
- Detección de régimen (`Trend_Bull`, `Trend_Bear`, `Range`, `HighVolatility`, `ExtremeEvent`).
- Ensemble probabilístico (RF + XGBoost + LSTM mock) con pesos dinámicos por régimen.
- Motor de decisión con expectativa matemática + gating de confirmación estadística/liquidez/volatilidad: opera solo si `E_t > 0` y señales son consistentes.
- Gestión adaptativa de riesgo (ATR, sizing, kill-switch por drawdown, rachas).
- Ejecución automática con cola asíncrona y límites de órdenes por segundo.
- Control robusto de rate limits para evitar bloqueos de IP (token bucket + weight monitor + header usage + backoff + cooldown 418).
- API Gateway FastAPI (`/health`, `/metrics/rate-limit`).
- Backtesting con métricas cuantitativas (Sharpe, Sortino, Calmar, PF, Expectancy).

## Formalización implementada

- `X_t`: vector de features (RSI, EMA, MACD, ATR, z-score, skewness, kurtosis, imbalance, breakout, etc).
- `R_t = g(X_t)`: clasificador de régimen.
- `M = w_1 RF + w_2 XGBoost + w_3 LSTM` con `w_i = h(R_t)`.
- `Score_t in [0,1]` y reglas de decisión:
  - `Score >= 0.75`: LONG
  - `Score <= 0.25`: SHORT
  - intermedio: HOLD
- `E_t = P(up)*R_up - P(down)*R_down`; sólo se opera con `E_t > 0`.

## Estructura exacta solicitada

```text
trading_system/
├── app/
│   ├── main.py
│   ├── config/
│   ├── core/
│   ├── services/
│   │   ├── market_data/
│   │   ├── feature_engineering/
│   │   ├── regime_detection/
│   │   ├── decision_engine/
│   │   ├── risk_management/
│   │   ├── execution/
│   │   ├── sentiment/
│   │   ├── monitoring/
│   ├── models/
│   │   ├── ensemble/
│   │   ├── ml_models/
│   ├── database/
│   ├── backtesting/
├── tests/
├── docker/
├── scripts/
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Quickstart

```bash
cp .env.example .env
python -m pip install -r requirements.txt
python trading_system/app/main.py
```

### Backtesting rápido

```bash
python scripts/run_backtest.py
```

## Seguridad recomendada

- API keys sólo con permisos de trading.
- Whitelist IP obligatoria.
- Separación estricta dev/test/prod.
- Secretos sólo en entorno (no en git).
