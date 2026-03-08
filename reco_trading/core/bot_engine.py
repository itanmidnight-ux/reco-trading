from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from rich.live import Live

from reco_trading.config.settings import Settings
from reco_trading.core.state_machine import BotState
from reco_trading.data.market_stream import MarketStream
from reco_trading.database.repository import Repository
from reco_trading.exchange.binance_client import BinanceClient
from reco_trading.exchange.order_manager import OrderManager
from reco_trading.risk.position_manager import Position, PositionManager
from reco_trading.risk.risk_manager import RiskManager
from reco_trading.strategy.confidence_model import ConfidenceModel
from reco_trading.strategy.indicators import apply_indicators
from reco_trading.strategy.signal_engine import SignalEngine
from reco_trading.ui.dashboard import DashboardSnapshot, TerminalDashboard


class BotEngine:
    """Orchestrates market analysis, risk controls, trading and monitoring."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.state = BotState.STARTING
        self.client = BinanceClient(settings.binance_api_key, settings.binance_api_secret, settings.binance_testnet)
        self.order_manager = OrderManager(self.client, settings.symbol)
        self.market_stream = MarketStream(self.client, settings.symbol, settings.history_limit)
        self.repository = Repository(settings.postgres_dsn)
        self.signal_engine = SignalEngine()
        self.confidence_model = ConfidenceModel()
        self.risk_manager = RiskManager(settings.daily_loss_limit_fraction, settings.max_trades_per_day)
        self.position_manager = PositionManager()
        self.trades_today = 0
        self.day_marker = datetime.now(timezone.utc).date()
        self.snapshot = DashboardSnapshot(pair=settings.symbol.replace("/", ""), timeframe="5m / 15m")

    async def run(self) -> None:
        await self.repository.setup()
        with Live(TerminalDashboard().render(self.snapshot), refresh_per_second=2) as live:
            while True:
                try:
                    await self._tick(live)
                except KeyboardInterrupt:
                    break
                except Exception as exc:  # noqa: BLE001
                    self.state = BotState.ERROR
                    await self._log("ERROR", f"runtime_error={exc}")
                    await asyncio.sleep(self.settings.loop_sleep_seconds)

    async def _tick(self, live: Live) -> None:
        self._roll_day()
        self.state = BotState.SYNCING_MARKET
        await self.order_manager.sync_rules()

        self.state = BotState.ANALYZING
        frame5 = apply_indicators(await self.market_stream.fetch_frame(self.settings.primary_timeframe))
        frame15 = apply_indicators(await self.market_stream.fetch_frame(self.settings.confirmation_timeframe))
        last_candle = frame5.iloc[-1]
        price = float(last_candle["close"])
        self.snapshot.price = price
        await self.repository.record_market_candle(
            self.settings.symbol,
            self.settings.primary_timeframe,
            {
                "open": float(last_candle["open"]),
                "high": float(last_candle["high"]),
                "low": float(last_candle["low"]),
                "close": float(last_candle["close"]),
                "volume": float(last_candle["volume"]),
            },
        )

        if self.signal_engine.is_sideways(frame5):
            self.state = BotState.WAITING_SIGNAL
            await self._log("INFO", "sideways_market_detected")
            self._update_dashboard()
            live.update(TerminalDashboard().render(self.snapshot))
            await asyncio.sleep(self.settings.loop_sleep_seconds)
            return

        bundle = self.signal_engine.generate(frame5, frame15)
        side, confidence = self.confidence_model.evaluate(bundle)
        await self.repository.record_signal(
            self.settings.symbol,
            {
                "trend": bundle.trend,
                "momentum": bundle.momentum,
                "volume": bundle.volume,
                "volatility": bundle.volatility,
                "structure": bundle.structure,
            },
            confidence,
            side,
        )
        self.snapshot.trend = bundle.trend
        self.snapshot.signals = bundle.__dict__
        self.snapshot.confidence = confidence

        balance = await self._fetch_usdt_balance()
        daily_pnl = await self.repository.get_daily_pnl()
        self.snapshot.balance = balance
        self.snapshot.daily_pnl = daily_pnl
        self.snapshot.available_capital = balance * self.settings.max_trade_balance_fraction

        self.state = BotState.CHECKING_RISK
        risk = self.risk_manager.validate(
            balance=balance,
            daily_pnl=daily_pnl,
            trades_today=self.trades_today,
            confidence=confidence,
            confidence_threshold=self.settings.confidence_threshold,
        )
        if not risk.approved:
            self.state = BotState.PAUSED if risk.reason == "RISK_PAUSE" else BotState.WAITING_SIGNAL
            await self._log("WARNING", f"risk_blocked reason={risk.reason}")
            await self._monitor_positions(price)
            self._update_dashboard()
            live.update(TerminalDashboard().render(self.snapshot))
            await asyncio.sleep(self.settings.loop_sleep_seconds)
            return

        if self.position_manager.can_open(confidence):
            await self._open_trade(side, price, float(last_candle["atr"]))

        await self._monitor_positions(price)
        self._update_dashboard()
        live.update(TerminalDashboard().render(self.snapshot))
        await asyncio.sleep(self.settings.loop_sleep_seconds)

    async def _open_trade(self, side: str, price: float, atr: float) -> None:
        self.state = BotState.PLACING_ORDER
        capital = self.snapshot.balance * self.settings.max_trade_balance_fraction
        qty = self.order_manager.normalize_quantity(capital / price)
        if qty <= 0 or not self.order_manager.validate_notional(qty, price):
            await self._log("WARNING", "order_rejected_by_exchange_filters")
            return

        order = await self.client.create_market_order(self.settings.symbol, side.lower(), qty)
        entry = float(order.get("average") or order.get("price") or price)
        stop_loss = entry - (1.5 * atr)
        take_profit = entry + (2.0 * atr)

        trade = await self.repository.create_trade(
            symbol=self.settings.symbol,
            side=side,
            quantity=qty,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_id=str(order.get("id")),
        )
        self.position_manager.open(
            Position(
                trade_id=trade.id,
                side=side,
                quantity=qty,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr=atr,
            )
        )
        self.trades_today += 1
        self.state = BotState.ORDER_FILLED
        await self._log("INFO", f"order_filled side={side} qty={qty:.8f} entry={entry:.2f}")

    async def _monitor_positions(self, current_price: float) -> None:
        if not self.position_manager.positions:
            return
        self.state = BotState.MONITORING_POSITION
        for position in list(self.position_manager.positions):
            exit_reason = self.position_manager.check_exit(position, current_price)
            if not exit_reason:
                continue
            self.state = BotState.CLOSING_POSITION
            close_side = "sell" if position.side == "BUY" else "buy"
            order = await self.client.create_market_order(self.settings.symbol, close_side, position.quantity)
            exit_price = float(order.get("average") or order.get("price") or current_price)
            pnl = (exit_price - position.entry_price) * position.quantity
            await self.repository.close_trade(position.trade_id, exit_price, pnl, exit_reason)
            self.position_manager.close(position.trade_id)
            self.state = BotState.TAKE_PROFIT_HIT if "TAKE_PROFIT" in exit_reason else BotState.STOP_LOSS_HIT
            await self._log("INFO", f"position_closed reason={exit_reason} pnl={pnl:.4f}")

    async def _fetch_usdt_balance(self) -> float:
        payload = await self.client.fetch_balance()
        usdt = payload.get("USDT") or {}
        return float(usdt.get("free", 0.0))

    async def _log(self, level: str, message: str) -> None:
        log_fn = getattr(self.logger, level.lower(), self.logger.info)
        log_fn(message)
        await self.repository.record_log(level, self.state.value, message)

    def _roll_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.day_marker:
            self.day_marker = today
            self.trades_today = 0

    def _update_dashboard(self) -> None:
        self.snapshot.state = self.state.value
        if self.position_manager.positions:
            position = self.position_manager.positions[0]
            self.snapshot.open_position = position.side
            self.snapshot.entry_price = position.entry_price
            self.snapshot.stop_loss = position.trailing_stop or position.stop_loss
            self.snapshot.take_profit = position.take_profit
            self.snapshot.position_size = position.quantity
        else:
            self.snapshot.open_position = "NONE"
            self.snapshot.entry_price = 0.0
            self.snapshot.stop_loss = 0.0
            self.snapshot.take_profit = 0.0
            self.snapshot.position_size = 0.0
