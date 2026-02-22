from __future__ import annotations

import asyncio
import signal
import traceback
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from reco_trading.config.settings import get_settings
from reco_trading.core.execution_engine import ExecutionEngine
from reco_trading.core.feature_engine import FeatureEngine
from reco_trading.core.institutional_risk import InstitutionalRiskManager, RiskConfig
from reco_trading.core.market_data import MarketDataService
from reco_trading.core.market_regime import MarketRegimeDetector
from reco_trading.core.mean_reversion_model import MeanReversionModel
from reco_trading.core.momentum_model import MomentumModel
from reco_trading.execution.execution_firewall import ExecutionFirewall
from reco_trading.infra.binance_client import BinanceClient
from reco_trading.infra.database import Database
from reco_trading.monitoring.metrics import MetricsExporter, TradingMetrics
from reco_trading.ui.terminal_dashboard import TerminalDashboard, VisualSnapshot


# ============================================================
# POSITION STATE
# ============================================================

class PositionState(Enum):
    FLAT = "flat"
    LONG = "long"


# ============================================================
# RUNTIME STATE (REAL – NO CAPITAL HARDCODEADO)
# ============================================================

@dataclass(slots=True)
class RuntimeState:
    equity: float = 0.0
    daily_pnl: float = 0.0
    trades: int = 0
    winning_trades: int = 0
    rolling_returns: list[float] = field(default_factory=list)
    consecutive_cycle_errors: int = 0
    position_state: PositionState = PositionState.FLAT
    position_qty: float = 0.0
    entry_price: float = 0.0


# ============================================================
# SIGNAL ENGINE
# ============================================================

class SignalEngine:
    def __init__(self) -> None:
        self.feature_engine = FeatureEngine()
        self.momentum = MomentumModel()
        self.reversion = MeanReversionModel()

    def generate(self, ohlcv: pd.DataFrame) -> dict[str, Any]:
        feats = self.feature_engine.build(ohlcv)
        if feats.empty:
            raise ValueError("Features vacías")

        momentum = float(np.clip(self.momentum.predict_proba_up(feats), 0.0, 1.0))
        mean_rev = float(np.clip(1.0 - self.reversion.predict_reversion(feats), 0.0, 1.0))

        returns = feats["return"].tail(300).to_numpy(dtype=float)

        return {
            "model_scores": {
                "momentum": momentum,
                "mean_reversion": mean_rev,
            },
            "atr": float(feats.iloc[-1]["atr14"]),
            "returns": returns,
            "prices": feats["close"].tail(300),
        }


# ============================================================
# DECISION ENGINE (HOLD DEFAULT)
# ============================================================

class DecisionEngine:
    def __init__(self) -> None:
        self.buy_threshold = 0.80
        self.sell_threshold = 0.20
        self.min_edge = 0.15

    def decide(self, scores: dict[str, float], regime: str) -> tuple[str, float]:
        mom = scores["momentum"]
        rev = scores["mean_reversion"]

        if regime == "trend":
            score = 0.7 * mom + 0.3 * rev
        else:
            score = 0.5 * mom + 0.5 * rev

        edge = score - 0.5

        if score >= self.buy_threshold and edge >= self.min_edge:
            return "BUY", score
        if score <= self.sell_threshold and abs(edge) >= self.min_edge:
            return "SELL", score

        return "HOLD", score


# ============================================================
# QUANT KERNEL (CORE)
# ============================================================

class QuantKernel:
    MAX_CONSECUTIVE_CYCLE_ERRORS = 5
    MIN_SECONDS_BETWEEN_TRADES = 300  # 5 minutos

    def __init__(self) -> None:
        self.s = get_settings()
        self.state = RuntimeState()
        self.last_trade_ts: float | None = None

        self.metrics = TradingMetrics()
        self.metrics_exporter = MetricsExporter(
            port=self.s.monitoring_metrics_port,
            addr=self.s.monitoring_metrics_host,
        )

        self.dashboard = TerminalDashboard()
        self.shutdown_event = asyncio.Event()
        self._shutdown_reason = "running"

    # --------------------------------------------------------

    async def initialize(self) -> None:
        self.client = BinanceClient(
            self.s.binance_api_key.get_secret_value(),
            self.s.binance_api_secret.get_secret_value(),
            testnet=self.s.binance_testnet,
            confirm_mainnet=self.s.confirm_mainnet,
        )

        self.db = Database(self.s.postgres_dsn, self.s.postgres_admin_dsn)

        self.market_data = MarketDataService(
            self.client, self.s.symbol, self.s.timeframe
        )

        self.signal_engine = SignalEngine()
        self.decision_engine = DecisionEngine()
        self.regime_detector = MarketRegimeDetector(n_states=3)

        self.risk_manager = InstitutionalRiskManager(
            RiskConfig(
                risk_per_trade=0.005,
                max_daily_loss=0.03,
                max_drawdown=0.2,
                max_exposure=0.1,
                max_correlation=0.7,
                kelly_fraction=0.5,
                max_consecutive_losses=5,
            )
        )

        self.execution_engine = ExecutionEngine(
            self.client,
            self.s.symbol,
            self.db,
            redis_url=self.s.redis_url,
            order_timeout_seconds=self.s.execution_order_timeout_seconds,
            firewall=ExecutionFirewall(
                max_total_exposure=100_000,
                max_asset_exposure=50_000,
                max_daily_loss=5_000,
                max_daily_notional=200_000,
            ),
            quant_kernel=self,
        )

        await self.client.ping()

        balance = await self.client.fetch_balance()
        usdt = float((balance.get("USDT") or {}).get("free") or 0.0)

        self.state.equity = usdt
        logger.info(f"Balance REAL sincronizado: {usdt:.2f} USDT")

    # --------------------------------------------------------

    async def run(self) -> None:
        await self.initialize()
        await self.db.init()
        self.metrics_exporter.start()
        self.dashboard.start()
        self._install_signal_handlers()

        while not self.shutdown_event.is_set():
            try:
                now = datetime.now(timezone.utc)
                ohlcv = await self.market_data.latest_ohlcv(limit=300)
                sig = self.signal_engine.generate(ohlcv)

                regime = self.regime_detector.predict(
                    sig["returns"], sig["prices"]
                ).get("regime", "range")

                decision, score = self.decision_engine.decide(
                    sig["model_scores"], regime
                )

                # ------------------------------------------
                # BLOQUEOS CRÍTICOS
                # ------------------------------------------

                if self.state.position_state == PositionState.LONG and decision == "BUY":
                    decision = "HOLD"

                if self.state.position_state == PositionState.FLAT and decision == "SELL":
                    decision = "HOLD"

                if self.last_trade_ts:
                    if (now.timestamp() - self.last_trade_ts) < self.MIN_SECONDS_BETWEEN_TRADES:
                        decision = "HOLD"

                # ------------------------------------------

                last_price = float(sig["prices"].iloc[-1])

                if decision == "BUY":
                    risk_amount = self.state.equity * 0.005
                    qty = risk_amount / last_price

                    fill = await self.execution_engine.execute("BUY", qty)
                    if fill:
                        self.state.position_state = PositionState.LONG
                        self.state.position_qty = qty
                        self.state.entry_price = last_price
                        self.last_trade_ts = now.timestamp()

                elif decision == "SELL":
                    qty = self.state.position_qty
                    fill = await self.execution_engine.execute("SELL", qty)
                    if fill:
                        pnl = (last_price - self.state.entry_price) * qty
                        self.state.daily_pnl += pnl
                        self.state.equity += pnl
                        self.state.position_state = PositionState.FLAT
                        self.state.position_qty = 0.0
                        self.state.entry_price = 0.0
                        self.last_trade_ts = now.timestamp()

                self.dashboard.update(
                    VisualSnapshot(
                        capital=self.state.equity,
                        balance=self.state.equity,
                        pnl_total=self.state.daily_pnl,
                        pnl_diario=self.state.daily_pnl,
                        drawdown=0.0,
                        riesgo_activo=0.5,
                        exposicion=self.state.position_qty * last_price,
                        trades=self.state.trades,
                        win_rate=0.0,
                        expectancy=0.0,
                        sharpe_rolling=0.0,
                        regimen=regime,
                        senal=decision,
                        latencia_ms=0.0,
                        ultimo_precio=last_price,
                        estado_binance="OK",
                        estado_sistema="OK",
                    )
                )

                await asyncio.sleep(self.s.loop_interval_seconds)

            except Exception as e:
                logger.error("kernel_error", error=str(e))
                traceback.print_exc()
                await asyncio.sleep(1)

    # --------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.shutdown_event.set)
