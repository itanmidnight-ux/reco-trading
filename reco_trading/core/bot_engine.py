from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

import ccxt
import pandas as pd
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
from reco_trading.strategy.signal_engine import SignalBundle, SignalEngine
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
        self.last_close_time: datetime | None = None
        self.trade_timestamps: deque[datetime] = deque()
        self.consecutive_losses = 0
        self.pause_trading_until: datetime | None = None

        self.snapshot = DashboardSnapshot(pair=settings.symbol.replace("/", ""), timeframe="5m / 15m")

    async def run(self) -> None:
        await self.initialize()
        with Live(TerminalDashboard().render(self.snapshot), refresh_per_second=2, transient=False) as live:
            while True:
                try:
                    self._roll_day()
                    market_data = await self.fetch_market_data()
                    analysis = self.analyze_market(market_data)

                    bundle: SignalBundle = analysis["bundle"]
                    side, confidence, grade = self.confidence_model.evaluate(bundle)
                    analysis["side"] = side
                    analysis["confidence"] = confidence
                    analysis["grade"] = grade
                    await self._persist_signal(bundle, side, confidence)

                    self._update_dashboard_from_analysis(market_data, analysis)

                    if await self.validate_trade_conditions(market_data, analysis):
                        size = self.calculate_position_size(analysis, market_data)
                        await self.execute_trade(analysis["side"], size, analysis)

                    await self.manage_open_position(market_data)
                    await self._refresh(live)
                except KeyboardInterrupt:
                    break
                except Exception as exc:  # noqa: BLE001
                    await self._set_state(BotState.ERROR, "runtime_error")
                    await self._log("ERROR", f"runtime_error={exc}")
                    await self.repository.record_error(self.state.value, "runtime", str(exc))
                    await asyncio.sleep(self.settings.loop_sleep_seconds)

    async def initialize(self) -> None:
        await self.repository.setup()
        await self.client.sync_time()
        await self.order_manager.sync_rules()
        await self._set_state(BotState.SYNCING_MARKET, "initialized")

    async def fetch_market_data(self) -> dict[str, Any]:
        await self._set_state(BotState.ANALYZING)
        frame5 = apply_indicators(await self.market_stream.fetch_frame(self.settings.primary_timeframe))
        frame15 = apply_indicators(await self.market_stream.fetch_frame(self.settings.confirmation_timeframe))
        ticker = await self.client.fetch_ticker(self.settings.symbol)
        order_book = await self.client.fetch_order_book(self.settings.symbol)

        last_candle = frame5.iloc[-1]
        price = float(ticker.get("last") or last_candle["close"])
        best_bid = float(order_book.get("bids", [[price]])[0][0])
        best_ask = float(order_book.get("asks", [[price]])[0][0])

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

        return {
            "frame5": frame5,
            "frame15": frame15,
            "price": price,
            "bid": best_bid,
            "ask": best_ask,
            "timestamp": datetime.now(timezone.utc),
        }

    def analyze_market(self, market_data: dict[str, Any]) -> dict[str, Any]:
        frame5: pd.DataFrame = market_data["frame5"]
        frame15: pd.DataFrame = market_data["frame15"]
        bundle = self.signal_engine.generate(frame5, frame15)
        candle = frame5.iloc[-1]
        htf = frame15.iloc[-1]

        return {
            "bundle": bundle,
            "atr": float(candle["atr"]),
            "adx": float(candle["adx"]),
            "volume_ratio": float(candle["volume"] / candle["vol_ma20"]) if float(candle["vol_ma20"]) > 0 else 0.0,
            "htf_bullish": bool(htf["ema20"] > htf["ema50"]),
            "htf_bearish": bool(htf["ema20"] < htf["ema50"]),
            "last_close": float(candle["close"]),
        }

    async def validate_trade_conditions(self, market_data: dict[str, Any], analysis: dict[str, Any]) -> bool:
        side = analysis["side"]
        confidence = float(analysis["confidence"])
        bundle: SignalBundle = analysis["bundle"]
        price = float(market_data["price"])
        spread_ratio = (float(market_data["ask"]) - float(market_data["bid"])) / max(price, 1e-9)

        self.snapshot.price = price
        balance = await self._fetch_usdt_balance()
        daily_pnl = await self.repository.get_daily_pnl()
        self.snapshot.balance = balance
        self.snapshot.available_capital = balance * self.settings.max_trade_balance_fraction
        self.snapshot.daily_pnl = daily_pnl

        if self.pause_trading_until and datetime.now(timezone.utc) < self.pause_trading_until:
            await self._set_state(BotState.PAUSED, "loss_protection_pause")
            await self._log("WARNING", f"trading_paused_until={self.pause_trading_until.isoformat()}")
            return False

        risk = self.risk_manager.validate(
            balance=balance,
            daily_pnl=daily_pnl,
            trades_today=self.trades_today,
            confidence=confidence,
            confidence_threshold=self.settings.confidence_threshold,
        )
        if not risk.approved:
            await self._set_state(BotState.PAUSED if risk.reason == "RISK_PAUSE" else BotState.WAITING_SIGNAL, risk.reason)
            await self._log("INFO", f"blocked_by_risk reason={risk.reason}")
            return False

        if self.signal_engine.is_sideways(market_data["frame5"]):
            await self._set_state(BotState.WAITING_SIGNAL, "sideways_market")
            return False

        if not bundle.regime_trade_allowed:
            await self._set_state(BotState.WAITING_SIGNAL, "low_volatility_regime")
            return False

        if spread_ratio > self.settings.max_spread_ratio:
            await self._log("INFO", f"blocked_by_spread spread_ratio={spread_ratio:.6f}")
            return False

        if float(analysis["volume_ratio"]) < self.settings.min_volume_ratio:
            await self._log("INFO", f"blocked_by_volume volume_ratio={analysis['volume_ratio']:.4f}")
            return False

        if float(analysis["adx"]) < self.settings.adx_min_threshold:
            await self._log("INFO", f"blocked_by_adx adx={analysis['adx']:.2f}")
            return False

        if side == "BUY" and not analysis["htf_bullish"]:
            await self._log("INFO", "blocked_by_htf_confirmation")
            return False
        if side == "SELL" and not analysis["htf_bearish"]:
            await self._log("INFO", "blocked_by_htf_confirmation")
            return False

        if not self._is_cooldown_complete():
            await self._log("INFO", "blocked_by_trade_cooldown")
            return False

        self._cleanup_trade_timestamps()
        if len(self.trade_timestamps) >= self.settings.max_trades_per_hour:
            await self._log("INFO", "blocked_by_hourly_trade_limit")
            return False

        return self.position_manager.can_open(confidence)

    def calculate_position_size(self, analysis: dict[str, Any], market_data: dict[str, Any]) -> float:
        balance = self.snapshot.balance
        price = float(market_data["price"])
        atr = max(float(analysis["atr"]), 1e-9)
        stop_distance = 1.5 * atr
        risk_per_trade = balance * self.settings.risk_per_trade_fraction

        raw_qty = risk_per_trade / stop_distance
        cap_qty = (self.snapshot.available_capital * analysis["bundle"].size_multiplier) / max(price, 1e-9)
        qty = min(raw_qty, cap_qty)
        qty = self.order_manager.normalize_quantity(qty)

        if qty <= 0:
            return 0.0
        if not self.order_manager.validate_notional(qty, price):
            return 0.0
        return qty

    async def execute_trade(self, side: str, quantity: float, analysis: dict[str, Any]) -> None:
        if quantity <= 0:
            await self._log("WARNING", "position_size_invalid")
            return

        await self._set_state(BotState.PLACING_ORDER)
        price = self.snapshot.price
        atr = float(analysis["atr"])

        try:
            order = await self.client.create_market_order(self.settings.symbol, side.lower(), quantity)
        except ccxt.BaseError as exc:
            await self._log("ERROR", f"order_rejected error={exc}")
            await self.repository.record_error(self.state.value, "order", str(exc))
            return

        entry = float(order.get("average") or order.get("price") or price)
        if side == "BUY":
            stop_loss = self.order_manager.normalize_price(entry - (1.5 * atr))
            take_profit = self.order_manager.normalize_price(entry + (2.0 * atr))
        else:
            stop_loss = self.order_manager.normalize_price(entry + (1.5 * atr))
            take_profit = self.order_manager.normalize_price(entry - (2.0 * atr))

        trade = await self.repository.create_trade(
            symbol=self.settings.symbol,
            side=side,
            quantity=quantity,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_id=str(order.get("id")),
        )

        self.position_manager.open(
            Position(
                trade_id=trade.id,
                side=side,
                quantity=quantity,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr=atr,
            )
        )

        self.trades_today += 1
        self.trade_timestamps.append(datetime.now(timezone.utc))
        await self._set_state(BotState.ORDER_FILLED)
        await self._log("INFO", f"order_filled side={side} qty={quantity:.8f} entry={entry:.4f}")

    async def manage_open_position(self, market_data: dict[str, Any]) -> None:
        if not self.position_manager.positions:
            return

        price = float(market_data["price"])
        await self._set_state(BotState.MONITORING_POSITION)

        for position in list(self.position_manager.positions):
            exit_reason = self.position_manager.check_exit(position, price)
            if not exit_reason:
                continue

            await self._set_state(BotState.CLOSING_POSITION)
            close_side = "sell" if position.side == "BUY" else "buy"
            order = await self.client.create_market_order(self.settings.symbol, close_side, position.quantity)
            exit_price = float(order.get("average") or order.get("price") or price)

            if position.side == "BUY":
                pnl = (exit_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - exit_price) * position.quantity

            await self.repository.close_trade(position.trade_id, exit_price, pnl, exit_reason)
            self.position_manager.close(position.trade_id)
            self.last_close_time = datetime.now(timezone.utc)
            pause_triggered = self._update_loss_protection(pnl)
            if pause_triggered and self.pause_trading_until:
                await self._log("WARNING", f"loss_protection_activated consecutive_losses={self.consecutive_losses} pause_until={self.pause_trading_until.isoformat()}")

            next_state = BotState.TAKE_PROFIT_HIT if "TAKE_PROFIT" in exit_reason else BotState.STOP_LOSS_HIT
            await self._set_state(next_state)
            await self._log("INFO", f"position_closed reason={exit_reason} pnl={pnl:.4f}")

    def _update_loss_protection(self, pnl: float) -> bool:
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= self.settings.loss_pause_after_consecutive:
            self.pause_trading_until = datetime.now(timezone.utc) + timedelta(minutes=self.settings.loss_pause_minutes)
            self.consecutive_losses = 0
            return True
        return False

    async def _persist_signal(self, bundle: SignalBundle, side: str, confidence: float) -> None:
        await self.repository.record_signal(
            self.settings.symbol,
            {
                "trend": bundle.trend,
                "momentum": bundle.momentum,
                "volume": bundle.volume,
                "volatility": bundle.volatility,
                "structure": bundle.structure,
                "order_flow": bundle.order_flow,
                "regime": bundle.regime,
            },
            confidence,
            side,
        )

    def _update_dashboard_from_analysis(self, market_data: dict[str, Any], analysis: dict[str, Any]) -> None:
        bundle: SignalBundle = analysis["bundle"]
        self.snapshot.state = self.state.value
        self.snapshot.price = float(market_data["price"])
        self.snapshot.trend = bundle.trend
        self.snapshot.signals = {
            "trend": bundle.trend,
            "momentum": bundle.momentum,
            "volume": bundle.volume,
            "volatility": bundle.volatility,
            "structure": bundle.structure,
            "order_flow": bundle.order_flow,
        }
        self.snapshot.confidence = float(analysis["confidence"])
        self.snapshot.signal_grade = str(analysis["grade"])
        self.snapshot.volatility_regime = bundle.regime
        self.snapshot.order_flow_signal = bundle.order_flow

    async def _fetch_usdt_balance(self) -> float:
        payload = await self.client.fetch_balance()
        usdt = payload.get("USDT") or {}
        return float(usdt.get("free", 0.0))

    async def _set_state(self, new_state: BotState, context: str = "") -> None:
        if new_state == self.state:
            return
        previous = self.state
        self.state = new_state
        await self.repository.record_state_change(previous.value, new_state.value, context)

    async def _log(self, level: str, message: str) -> None:
        getattr(self.logger, level.lower(), self.logger.info)(message)
        await self.repository.record_log(level, self.state.value, message)

    def _roll_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.day_marker:
            self.day_marker = today
            self.trades_today = 0

    def _is_cooldown_complete(self) -> bool:
        if self.last_close_time is None:
            return True
        return datetime.now(timezone.utc) - self.last_close_time >= timedelta(minutes=self.settings.cooldown_minutes)

    def _cleanup_trade_timestamps(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        while self.trade_timestamps and self.trade_timestamps[0] < cutoff:
            self.trade_timestamps.popleft()

    async def _refresh(self, live: Live) -> None:
        self._update_dashboard_position()
        live.update(TerminalDashboard().render(self.snapshot), refresh=True)
        await asyncio.sleep(self.settings.loop_sleep_seconds)

    def _update_dashboard_position(self) -> None:
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
