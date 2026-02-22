from __future__ import annotations

import asyncio
from contextlib import suppress
import signal
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from reco_trading.core.portfolio_optimization import ConvexPortfolioOptimizer
from reco_trading.execution.execution_firewall import ExecutionFirewall
from reco_trading.infra.binance_client import BinanceClient
from reco_trading.infra.database import Database
from reco_trading.kernel.capital_governor import CapitalGovernor
from reco_trading.monitoring.metrics import MetricsExporter, TradingMetrics
from reco_trading.ui.terminal_dashboard import TerminalDashboard, VisualSnapshot


# =========================
# RUNTIME STATE CORREGIDO
# =========================
@dataclass(slots=True)
class RuntimeState:
    equity: float = 0.0
    daily_pnl: float = 0.0
    trades: int = 0
    winning_trades: int = 0
    rolling_returns: list[float] = field(default_factory=list)
    consecutive_cycle_errors: int = 0
    last_trade_ts: float = 0.0


class QuantKernel:
    MAX_CONSECUTIVE_CYCLE_ERRORS = 5
    MIN_SECONDS_BETWEEN_TRADES = 90  # ⛔ anti-overtrading
    MIN_SIGNAL_CONFIDENCE = 0.60     # ⛔ filtro de calidad

    def __init__(self) -> None:
        self.s = get_settings()
        self.metrics = TradingMetrics()
        self.metrics_exporter = MetricsExporter(
            port=self.s.monitoring_metrics_port,
            addr=self.s.monitoring_metrics_host,
        )

        self.state = RuntimeState()
        self.initial_equity = 0.0
        self.blocked = False

        self.dashboard = TerminalDashboard()
        self.shutdown_event = asyncio.Event()
        self._shutdown_reason = 'running'

    # =========================
    # INITIALIZATION
    # =========================
    async def initialize(self) -> None:
        api_key = self.s.binance_api_key.get_secret_value().strip()
        api_secret = self.s.binance_api_secret.get_secret_value().strip()

        if not api_key or not api_secret:
            raise RuntimeError("Missing Binance credentials")

        self.client = BinanceClient(
            api_key,
            api_secret,
            testnet=self.s.binance_testnet,
            confirm_mainnet=self.s.confirm_mainnet,
        )

        self.db = Database(self.s.postgres_dsn, self.s.postgres_admin_dsn)

        self.market_data = MarketDataService(self.client, self.s.symbol, self.s.timeframe)

        self.signal_engine = SignalEngine()
        self.regime_detector = MarketRegimeDetector(n_states=3)
        self.decision_engine = DecisionEngine(
            buy_threshold=0.62,
            sell_threshold=0.38,
            min_edge=0.04,
        )

        self.risk_manager = InstitutionalRiskManager(
            RiskConfig(
                risk_per_trade=0.002,   # ⛔ REDUCIDO
                max_daily_loss=0.02,
                max_drawdown=0.15,
                max_exposure=0.15,
                max_correlation=0.8,
                kelly_fraction=0.4,
                max_consecutive_losses=3,
            )
        )

        self.capital_governor = CapitalGovernor(
            hard_cap_global=100_000,
            max_risk_per_trade_ratio=0.002,
            max_daily_loss_ratio=0.02,
        )

        self.execution_firewall = ExecutionFirewall(
            max_total_exposure=100_000,
            max_asset_exposure=50_000,
            max_daily_loss=3_000,
            max_daily_notional=150_000,
        )

        self.execution_engine = ExecutionEngine(
            self.client,
            self.s.symbol,
            self.db,
            redis_url=self.s.redis_url,
            firewall=self.execution_firewall,
            quant_kernel=self,
            capital_governor=self.capital_governor,
        )

        logger.info("Validando conexión Binance...")
        await self.client.ping()

        ticker = await self.client.fetch_ticker(self.s.symbol)
        logger.info(f"Precio inicial BTC/USDT: {ticker['last']}")

        # 🔥 CAPITAL REAL
        balance = await self.client.fetch_balance()
        usdt_balance = float((balance.get('USDT') or {}).get('free') or 0.0)

        if usdt_balance <= 0:
            raise RuntimeError("Balance USDT inválido")

        self.state.equity = usdt_balance
        self.initial_equity = usdt_balance

        logger.success(f"Capital REAL sincronizado: {usdt_balance:.2f} USDT")

    # =========================
    # MAIN LOOP
    # =========================
    async def run(self) -> None:
        await self.initialize()
        await self.db.init()
        self.metrics_exporter.start()
        self.dashboard.start()
        self._install_signal_handlers()

        while not self.shutdown_event.is_set():
            try:
                now = datetime.now(timezone.utc).timestamp()

                if now - self.state.last_trade_ts < self.MIN_SECONDS_BETWEEN_TRADES:
                    await asyncio.sleep(2)
                    continue

                ohlcv = await self.market_data.latest_ohlcv(limit=300)
                sig = self.signal_engine.generate(ohlcv)

                regime = self.regime_detector.predict(sig["returns"], sig["prices"])
                decision, score = self.decision_engine.decide(sig["model_scores"], regime["regime"])

                if score < self.MIN_SIGNAL_CONFIDENCE or decision == "HOLD":
                    await asyncio.sleep(self.s.loop_interval_seconds)
                    continue

                self.risk_manager.update_equity(self.state.equity)
                self.risk_manager.check_kill_switch(self.state.equity)

                if self.risk_manager.kill_switch:
                    self.blocked = True
                    self.request_shutdown("risk_kill_switch")
                    return

                last_price = float(sig["prices"].iloc[-1])
                position_size = min(
                    self.state.equity * 0.01 / last_price,
                    0.002
                )

                fill = await self.execution_engine.execute(decision, position_size)
                if not fill:
                    await asyncio.sleep(2)
                    continue

                realized = float(fill.get("pnl", 0.0))
                self.state.daily_pnl += realized
                self.state.equity += realized
                self.state.trades += 1
                self.state.last_trade_ts = now

                if realized > 0:
                    self.state.winning_trades += 1

                self.state.rolling_returns.append(realized)
                self.state.rolling_returns = self.state.rolling_returns[-100:]

                logger.success(
                    f"TRADE {decision} | pnl={realized:.2f} | equity={self.state.equity:.2f}"
                )

                await asyncio.sleep(self.s.loop_interval_seconds)

            except Exception as e:
                logger.error("kernel_error", error=str(e))
                await asyncio.sleep(2)

    def request_shutdown(self, reason: str) -> None:
        self._shutdown_reason = reason
        self.shutdown_event.set()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(Exception):
                loop.add_signal_handler(sig, self.request_shutdown, sig.name)
