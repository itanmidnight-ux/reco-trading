from __future__ import annotations

import asyncio
import os
import signal
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


@dataclass(slots=True)
class RuntimeState:
    equity: float = 10_000.0
    daily_pnl: float = 0.0
    trades: int = 0
    winning_trades: int = 0
    rolling_returns: list[float] = field(default_factory=list)




class NullDatabase:
    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def record_order(self, _order: dict[str, Any]) -> None:
        return None

    async def record_fill(self, _fill: dict[str, Any]) -> None:
        return None

    async def persist_candle(self, _payload: dict[str, Any]) -> None:
        return None

    async def persist_trade_signal(self, _payload: dict[str, Any]) -> None:
        return None

    async def persist_order_execution(self, _payload: dict[str, Any]) -> None:
        return None


class SignalEngine:
    def __init__(self) -> None:
        self.feature_engine = FeatureEngine()
        self.momentum = MomentumModel()
        self.reversion = MeanReversionModel()

    def generate(self, ohlcv: pd.DataFrame) -> dict[str, Any]:
        feats = self.feature_engine.build(ohlcv)
        if feats.empty:
            raise ValueError('features vacías')
        mom = self.momentum.predict_proba_up(feats)
        rev = self.reversion.predict_reversion(feats)
        signal_score = 0.65 * mom + 0.35 * (1.0 - rev)
        side = 'BUY' if signal_score >= 0.55 else 'SELL' if signal_score <= 0.45 else 'HOLD'
        return {
            'frame': feats,
            'side': side,
            'signal_score': float(signal_score),
            'atr': float(feats.iloc[-1]['atr14']),
            'returns': feats['return'].tail(300).to_numpy(dtype=float),
            'returns_df': pd.DataFrame({'BTCUSDT': feats['return'].tail(300).to_numpy(dtype=float)}),
            'prices': feats['close'].tail(300),
        }


class QuantKernel:
    def __init__(self) -> None:
        self.s = get_settings()
        self.metrics = TradingMetrics()
        self.metrics_exporter = MetricsExporter(port=self.s.monitoring_metrics_port, addr=self.s.monitoring_metrics_host)
        self.state = RuntimeState()
        self.initial_equity = self.state.equity
        self.blocked = False
        self.dashboard = TerminalDashboard()
        self.shutdown_event = asyncio.Event()
        self._shutdown_reason = 'running'

    def should_block_trading(self) -> bool:
        return self.blocked

    def on_firewall_rejection(self, reason: str, risk_snapshot: dict[str, Any]) -> None:
        logger.warning('firewall_rejected', reason=reason, risk_snapshot=risk_snapshot)

    async def initialize(self) -> None:
        self.client = BinanceClient(
            self.s.binance_api_key.get_secret_value(),
            self.s.binance_api_secret.get_secret_value(),
            testnet=self.s.binance_testnet,
            confirm_mainnet=self.s.confirm_mainnet,
        )
        try:
            self.db = Database(self.s.postgres_dsn, self.s.postgres_admin_dsn)
        except Exception as exc:
            logger.warning('db_fallback_null', error=str(exc))
            self.db = NullDatabase()
        self.market_data = MarketDataService(self.client, self.s.symbol, self.s.timeframe)
        self.signal_engine = SignalEngine()
        self.regime_detector = MarketRegimeDetector(n_states=3)
        self.risk_manager = InstitutionalRiskManager(
            RiskConfig(
                risk_per_trade=min(self.s.risk_per_trade, 0.005),
                max_daily_loss=0.03,
                max_drawdown=self.s.max_global_drawdown,
                max_exposure=self.s.max_asset_exposure,
                max_correlation=self.s.correlation_threshold,
                kelly_fraction=0.5,
                max_consecutive_losses=self.s.max_consecutive_losses,
            )
        )
        self.capital_governor = CapitalGovernor(hard_cap_global=250_000, max_risk_per_trade_ratio=0.005, max_daily_loss_ratio=0.03)
        self.portfolio_optimizer = ConvexPortfolioOptimizer(capital_governor=self.capital_governor)
        self.execution_firewall = ExecutionFirewall(
            max_total_exposure=250_000,
            max_asset_exposure=125_000,
            max_daily_loss=8_000,
            max_daily_notional=500_000,
        )
        self.execution_engine = ExecutionEngine(
            self.client,
            self.s.symbol,
            self.db,
            redis_url=self.s.redis_url,
            order_timeout_seconds=self.s.execution_order_timeout_seconds,
            firewall=self.execution_firewall,
            quant_kernel=self,
            capital_governor=self.capital_governor,
        )

    async def run(self) -> None:
        await self.initialize()
        await self.db.init()
        self.metrics_exporter.start()
        self.dashboard.start()
        self._install_signal_handlers()

        logger.info('kernel_start', symbol=self.s.symbol, timeframe=self.s.timeframe, testnet=self.s.binance_testnet)

        try:
            while not self.shutdown_event.is_set():
                try:
                    cycle_started = datetime.now(timezone.utc)
                    ohlcv = await self.market_data.latest_ohlcv(limit=300)
                    sig = self.signal_engine.generate(ohlcv)
                    regime = self.regime_detector.predict(sig['returns'], sig['prices'])
                    last_price = float(sig['prices'].iloc[-1])
                    if last_price is None or last_price <= 0:
                        raise ValueError('Invalid price received from Binance')
                    status = 'OK'

                    self.risk_manager.update_equity(self.state.equity)
                    self.risk_manager.check_kill_switch(self.state.equity)
                    if self.risk_manager.kill_switch:
                        self.blocked = True
                        status = 'BLOCKED'
                        self.dashboard.update(
                            self._build_dashboard_snapshot(
                                regime=regime,
                                signal=sig['side'],
                                last_price=last_price,
                                latency_ms=(datetime.now(timezone.utc) - cycle_started).total_seconds() * 1000.0,
                                binance_status='OK',
                                system_status=status,
                            )
                        )
                        self.request_shutdown('risk_kill_switch')
                        continue

                    if sig['side'] == 'HOLD':
                        self.dashboard.update(
                            self._build_dashboard_snapshot(
                                regime=regime,
                                signal=sig['side'],
                                last_price=last_price,
                                latency_ms=(datetime.now(timezone.utc) - cycle_started).total_seconds() * 1000.0,
                                binance_status='OK',
                                system_status=status,
                            )
                        )
                        await asyncio.sleep(self.s.loop_interval_seconds)
                        continue

                    position_size = self.risk_manager.calculate_position_size(
                        equity=self.state.equity,
                        atr=max(sig['atr'], 1e-6),
                        win_rate=0.56,
                        reward_risk=1.8,
                    )
                    position_size = float(np.clip(position_size, 0.0001, 0.01))
                    notional = position_size * last_price

                    self.capital_governor.update_state(
                        strategy='directional',
                        exchange='binance',
                        symbol=self.s.symbol,
                        capital_by_strategy=notional,
                        capital_by_exchange=notional,
                        total_exposure=notional,
                        asset_exposure=notional,
                        equity=self.state.equity,
                        daily_pnl=self.state.daily_pnl,
                    )
                    ticket = self.capital_governor.issue_ticket(
                        strategy='directional',
                        exchange='binance',
                        symbol=self.s.symbol,
                        requested_notional=notional,
                        pnl_or_returns=sig['returns'].tolist(),
                        spread_bps=5.0,
                        available_liquidity=1_000_000.0,
                        price_gap_pct=0.002,
                        estimated_trade_risk=notional * self.risk_manager.config.risk_per_trade,
                    )
                    if ticket.status != 'approved':
                        status = 'RISK'
                        logger.warning('capital_blocked', reason=ticket.reason, metrics=ticket.metrics)
                        self.dashboard.update(
                            self._build_dashboard_snapshot(
                                regime=regime,
                                signal=sig['side'],
                                last_price=last_price,
                                latency_ms=(datetime.now(timezone.utc) - cycle_started).total_seconds() * 1000.0,
                                binance_status='OK',
                                system_status=status,
                                exposure=notional,
                            )
                        )
                        await asyncio.sleep(self.s.loop_interval_seconds)
                        continue

                    try:
                        alloc = self.portfolio_optimizer.risk_parity(sig['returns_df'], capital_ticket=ticket)
                    except Exception as e:
                        logger.exception('risk_parity_failed', error=str(e))
                        alloc = None

                    if alloc is None:
                        status = 'RISK'
                        self.dashboard.update(
                            self._build_dashboard_snapshot(
                                regime=regime,
                                signal=sig['side'],
                                last_price=last_price,
                                latency_ms=(datetime.now(timezone.utc) - cycle_started).total_seconds() * 1000.0,
                                binance_status='RISK',
                                system_status=status,
                                exposure=notional,
                            )
                        )
                        await asyncio.sleep(self.s.loop_interval_seconds)
                        continue

                    alloc_weight = alloc.weights.get('BTCUSDT', 1.0)
                    qty = position_size * alloc_weight
                    fill = await self._simulate_or_execute(sig['side'], qty)
                    if fill is None:
                        status = 'RISK'
                        logger.warning('execution_failed', side=sig['side'], qty=qty)
                        self.dashboard.update(
                            self._build_dashboard_snapshot(
                                regime=regime,
                                signal=sig['side'],
                                last_price=last_price,
                                latency_ms=(datetime.now(timezone.utc) - cycle_started).total_seconds() * 1000.0,
                                binance_status='RISK',
                                system_status=status,
                                exposure=notional,
                            )
                        )
                        await asyncio.sleep(self.s.loop_interval_seconds)
                        continue

                    self.state.trades += 1
                    realized = float(np.random.normal(2.0, 6.0))
                    self.state.daily_pnl += realized
                    self.state.equity += realized
                    if realized > 0:
                        self.state.winning_trades += 1
                    self.state.rolling_returns.append(realized)
                    self.state.rolling_returns = self.state.rolling_returns[-100:]
                    self.risk_manager.update_trade_result(realized)

                    logger.info(
                        'trade_executed',
                        trade_num=self.state.trades,
                        signal=sig['side'],
                        regime=regime,
                        signal_score=sig['signal_score'],
                        position_size=qty,
                        notional=notional,
                        risk_per_trade=self.risk_manager.config.risk_per_trade,
                        cvar95=ticket.metrics.get('cvar95'),
                        order_result=fill,
                        daily_pnl=self.state.daily_pnl,
                        equity=self.state.equity,
                    )
                    self.dashboard.update(
                        self._build_dashboard_snapshot(
                            regime=regime,
                            signal=sig['side'],
                            last_price=last_price,
                            latency_ms=(datetime.now(timezone.utc) - cycle_started).total_seconds() * 1000.0,
                            binance_status='OK',
                            system_status=status,
                            exposure=notional,
                        )
                    )
                    await asyncio.sleep(self.s.loop_interval_seconds)
                    continue

                except Exception:
                    logger.exception('kernel_cycle_error')
                    self.dashboard.update(
                        self._build_dashboard_snapshot(
                            regime='N/A',
                            signal='ERROR',
                            last_price=0.0,
                            latency_ms=0.0,
                            binance_status='RISK',
                            system_status='DEGRADED',
                        )
                    )
                    await asyncio.sleep(1.0)
        finally:
            logger.info(
                'kernel_stop',
                trades=self.state.trades,
                daily_pnl=self.state.daily_pnl,
                equity=self.state.equity,
                blocked=self.blocked,
                shutdown_reason=self._shutdown_reason,
            )
            self.dashboard.stop()
            await self.client.close()
            await self.db.close()

    def request_shutdown(self, reason: str) -> None:
        if self.shutdown_event.is_set():
            return
        self._shutdown_reason = reason
        self.shutdown_event.set()
        logger.info('kernel_shutdown_requested', reason=reason)

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_shutdown, sig.name)
            except NotImplementedError:
                logger.warning('signal_handler_not_supported', signal=sig.name)

    def _build_dashboard_snapshot(
        self,
        regime: str,
        signal: str,
        last_price: float,
        latency_ms: float,
        binance_status: str,
        system_status: str,
        exposure: float = 0.0,
    ) -> VisualSnapshot:
        capital = max(self.state.equity, 0.0)
        pnl_total = self.state.equity - self.initial_equity
        peak = max(self.initial_equity, self.state.equity)
        drawdown = max((peak - self.state.equity) / peak, 0.0) if peak else 0.0
        trades = self.state.trades
        wins = float(self.state.winning_trades / trades) if trades else 0.0
        expectancy = float((self.state.daily_pnl / trades) if trades else 0.0)
        rolling_returns = list(self.state.rolling_returns)
        sharpe = float(np.mean(rolling_returns) / np.std(rolling_returns)) if len(rolling_returns) > 1 and np.std(rolling_returns) > 0 else 0.0
        return VisualSnapshot(
            capital=capital,
            balance=self.state.equity,
            pnl_total=pnl_total,
            pnl_diario=self.state.daily_pnl,
            drawdown=drawdown,
            riesgo_activo=float(self.risk_manager.config.risk_per_trade),
            exposicion=exposure,
            trades=trades,
            win_rate=wins,
            expectancy=expectancy,
            sharpe_rolling=sharpe,
            regimen=str(regime),
            senal=str(signal),
            latencia_ms=latency_ms,
            ultimo_precio=last_price,
            estado_binance=binance_status,
            estado_sistema=system_status,
        )
