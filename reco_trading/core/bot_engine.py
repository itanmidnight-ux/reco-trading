from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import ccxt
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from reco_trading.config.settings import Settings
from reco_trading.config.symbols import normalize_symbol
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
from reco_trading.ui.dashboard import TerminalDashboard


class BotEngine:
    """Orchestrates market analysis, risk controls, trading and monitoring."""

    def __init__(self, settings: Settings, state_manager: "StateManager | None" = None) -> None:
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.state = BotState.INITIALIZING

        self.client = BinanceClient(settings.binance_api_key, settings.binance_api_secret, settings.binance_testnet)
        self.symbol = normalize_symbol(settings.trading_symbol)
        self.order_manager = OrderManager(self.client, self.symbol)
        self.market_stream = MarketStream(self.client, self.symbol, settings.history_limit)
        self.repository = Repository(settings.postgres_dsn)
        self.signal_engine = SignalEngine()
        self.confidence_model = ConfidenceModel()
        self.risk_manager = RiskManager(settings.daily_loss_limit_fraction, settings.max_trades_per_day)
        self.position_manager = PositionManager()

        self.state_manager = state_manager
        self.trades_today = 0
        self.win_count = 0
        self.start_time = time.time()
        self.day_marker = datetime.now(timezone.utc).date()
        self.last_close_time: datetime | None = None
        self.pause_trading_until: datetime | None = None
        self.consecutive_losses = 0

        self.dashboard = TerminalDashboard()
        self.console = Console()
        self.snapshot: dict[str, Any] = {
            "pair": self.symbol,
            "timeframe": f"{self.settings.primary_timeframe} / {self.settings.confirmation_timeframe}",
            "price": None,
            "spread": None,
            "trend": None,
            "adx": None,
            "volatility_regime": None,
            "order_flow": None,
            "signal": None,
            "confidence": None,
            "balance": None,
            "equity": None,
            "daily_pnl": None,
            "trades_today": 0,
            "win_rate": None,
            "last_trade": None,
            "cooldown": None,
            "status": "INITIALIZING",
            "exchange_status": "UNKNOWN",
            "database_status": "UNKNOWN",
            "redis_status": "UNKNOWN",
            "signals": {},
            "volume": None,
            "api_latency_ms": None,
            "started_at": time.time(),
        }

    async def run(self) -> None:
        await self._set_state(BotState.INITIALIZING, "load_settings")
        await self.repository.setup()
        self.snapshot["database_status"] = "CONNECTED"
        await self._set_state(BotState.CONNECTING_EXCHANGE, "connect_binance")
        await self.client.sync_time()
        self.snapshot["exchange_status"] = "CONNECTED"
        await self._set_state(BotState.SYNCING_SYMBOL, "sync_symbol")
        await self._set_state(BotState.SYNCING_RULES, "sync_exchange_rules")
        await self.order_manager.sync_rules()
        self._sync_ui_state()

        with Live(self.dashboard.render(self.snapshot), refresh_per_second=2, transient=False) as live:
            while True:
                try:
                    self._roll_day()
                    await self._set_state(BotState.WAITING_MARKET_DATA)
                    market_data = await self.fetch_market_data()
                    await self._set_state(BotState.ANALYZING_MARKET)
                    analysis = await self.analyze_market(market_data)
                    self._update_snapshot(market_data, analysis)

                    if await self.validate_trade_conditions(analysis):
                        await self.execute_trade(analysis, market_data)

                    await self.manage_open_position(market_data)
                    self._sync_ui_state()
                    self._safe_live_update(live)
                    await asyncio.sleep(self.settings.loop_sleep_seconds)
                except KeyboardInterrupt:
                    await self._set_state(BotState.STOPPED, "manual_stop")
                    break
                except Exception as exc:  # noqa: BLE001
                    await self._set_state(BotState.ERROR, "runtime_error")
                    await self._log("ERROR", f"runtime_error={exc}")
                    await self.repository.record_error(self.state.value, "runtime", str(exc))
                    self.snapshot["status"] = BotState.ERROR.value
                    self._safe_live_update(live)
                    await asyncio.sleep(self.settings.loop_sleep_seconds)

    async def fetch_market_data(self) -> dict[str, Any]:
        frame5 = apply_indicators(await self.market_stream.fetch_frame(self.settings.primary_timeframe))
        frame15 = apply_indicators(await self.market_stream.fetch_frame(self.settings.confirmation_timeframe))
        candle = frame5.iloc[-1]

        tick_start = time.perf_counter()
        ticker = await self.client.fetch_ticker(self.symbol)
        order_book = await self.client.fetch_order_book(self.symbol)
        self.snapshot["api_latency_ms"] = (time.perf_counter() - tick_start) * 1000
        price = _as_float(ticker.get("last"), _as_float(candle.get("close"), 0.0))
        bid = _book_price(order_book, "bids", price)
        ask = _book_price(order_book, "asks", price)
        spread = max(ask - bid, 0.0)

        await self.repository.record_market_candle(
            self.symbol,
            self.settings.primary_timeframe,
            {
                "open": _as_float(candle.get("open"), price),
                "high": _as_float(candle.get("high"), price),
                "low": _as_float(candle.get("low"), price),
                "close": _as_float(candle.get("close"), price),
                "volume": _as_float(candle.get("volume"), 0.0),
            },
        )

        return {
            "frame5": frame5,
            "frame15": frame15,
            "candle": candle,
            "price": price,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "volume": _as_float(candle.get("volume"), 0.0),
            "atr": _as_float(candle.get("atr"), 0.0),
            "adx": _as_float(candle.get("adx"), 0.0),
            "change_24h": _as_float(ticker.get("percentage"), 0.0),
        }

    async def analyze_market(self, market_data: dict[str, Any]) -> dict[str, Any]:
        bundle: SignalBundle = self.signal_engine.generate(market_data["frame5"], market_data["frame15"])
        side, confidence, grade = self.confidence_model.evaluate(bundle)
        await self._set_state(BotState.SIGNAL_GENERATED)
        await self._persist_signal(bundle, side, confidence)
        return {"bundle": bundle, "side": side, "confidence": confidence, "grade": grade}

    async def validate_trade_conditions(self, analysis: dict[str, Any]) -> bool:
        confidence = float(analysis["confidence"])
        balance = await self._fetch_usdt_balance()
        daily_pnl = float(await self.repository.get_daily_pnl() or 0.0)

        self.snapshot["balance"] = balance
        self.snapshot["equity"] = balance + daily_pnl
        self.snapshot["daily_pnl"] = daily_pnl
        self.snapshot["trades_today"] = self.trades_today

        if self.pause_trading_until and datetime.now(timezone.utc) < self.pause_trading_until:
            await self._set_state(BotState.PAUSED, "loss_protection_pause")
            self.snapshot["cooldown"] = f"PAUSED until {self.pause_trading_until.isoformat(timespec='seconds')}"
            return False

        if not self._is_cooldown_complete():
            await self._set_state(BotState.COOLDOWN, "cooldown_active")
            self.snapshot["cooldown"] = "ACTIVE"
            return False

        risk = self.risk_manager.validate(
            balance=balance,
            daily_pnl=daily_pnl,
            trades_today=self.trades_today,
            confidence=confidence,
            confidence_threshold=self.settings.confidence_threshold,
        )
        if not risk.approved:
            await self._set_state(BotState.PAUSED if risk.reason == "RISK_PAUSE" else BotState.COOLDOWN, risk.reason)
            self.snapshot["cooldown"] = risk.reason
            return False

        self.snapshot["cooldown"] = "READY"
        return True

    async def execute_trade(self, analysis: dict[str, Any], market_data: dict[str, Any]) -> None:
        bundle: SignalBundle = analysis["bundle"]
        side = str(analysis["side"])
        price = float(market_data["price"])
        atr = float(market_data.get("atr", 0.0))

        if not bundle.regime_trade_allowed:
            await self._set_state(BotState.COOLDOWN, "regime_filter")
            return

        if not self.position_manager.can_open(float(analysis["confidence"])):
            await self._set_state(BotState.POSITION_OPEN, "max_positions")
            return

        qty = self.calculate_position_size(price, float(bundle.size_multiplier))
        if qty <= 0:
            await self._log("WARNING", "quantity_below_minimum")
            return
        if not self.order_manager.validate_notional(qty, price):
            await self._log("WARNING", "notional_below_minimum")
            return

        await self._set_state(BotState.PLACING_ORDER)
        try:
            order = await self.client.create_market_order(self.symbol, side.lower(), qty)
        except ccxt.BaseError as exc:
            await self._log("ERROR", f"order_rejected error={exc}")
            await self.repository.record_error(self.state.value, "order", str(exc))
            return

        entry = _as_float(order.get("average"), _as_float(order.get("price"), price))
        stop_loss, take_profit = self._build_stops(side, entry, atr)
        trade = await self.repository.create_trade(
            symbol=self.symbol,
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
                atr=max(atr, 0.0),
            )
        )

        self.trades_today += 1
        self.snapshot["trades_today"] = self.trades_today
        self.snapshot["last_trade"] = f"{side} @ {entry:.2f}"
        await self._set_state(BotState.POSITION_OPEN)
        await self._log("INFO", f"order_filled side={side} qty={qty:.8f} entry={entry:.2f}")

        if self.state_manager:
            self.state_manager.add_trade(
                {
                    "trade_id": trade.id,
                    "time": datetime.utcnow().isoformat(timespec="seconds"),
                    "pair": self.symbol,
                    "side": side,
                    "entry": entry,
                    "size": qty,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "status": "OPEN",
                    "entry_time": datetime.utcnow().isoformat(timespec="seconds"),
                    "exit_time": "-",
                    "fees": order.get("fee", {}).get("cost", 0),
                    "confidence": analysis.get("confidence"),
                    "signal_details": str(bundle),
                }
            )

    async def manage_open_position(self, market_data: dict[str, Any]) -> None:
        if not self.position_manager.positions:
            return

        price = float(market_data["price"])
        await self._set_state(BotState.POSITION_OPEN)
        for position in list(self.position_manager.positions):
            exit_reason = self.position_manager.check_exit(position, price)
            if not exit_reason:
                continue

            await self._set_state(BotState.COOLDOWN)
            close_side = "sell" if position.side == "BUY" else "buy"
            order = await self.client.create_market_order(self.symbol, close_side, position.quantity)
            exit_price = _as_float(order.get("average"), _as_float(order.get("price"), price))

            pnl = (exit_price - position.entry_price) * position.quantity
            if position.side == "SELL":
                pnl *= -1

            await self.repository.close_trade(position.trade_id, exit_price, pnl, exit_reason)
            self.position_manager.close(position.trade_id)
            self.last_close_time = datetime.now(timezone.utc)
            if self._update_loss_protection(pnl):
                await self._log("WARNING", "loss_protection_enabled")

            if pnl > 0:
                self.win_count += 1
            self.snapshot["win_rate"] = self.win_count / self.trades_today if self.trades_today else None
            self.snapshot["last_trade"] = f"{position.side} {exit_reason} pnl={pnl:.4f}"

            if self.state_manager:
                self.state_manager.add_trade(
                    {
                        "trade_id": position.trade_id,
                        "time": datetime.utcnow().isoformat(timespec="seconds"),
                        "pair": self.symbol,
                        "side": position.side,
                        "entry": position.entry_price,
                        "exit": exit_price,
                        "size": position.quantity,
                        "pnl": pnl,
                        "status": exit_reason,
                        "entry_time": "-",
                        "exit_time": datetime.utcnow().isoformat(timespec="seconds"),
                        "fees": order.get("fee", {}).get("cost", 0),
                        "confidence": None,
                        "signal_details": exit_reason,
                    }
                )

    def calculate_position_size(self, price: float, size_multiplier: float) -> float:
        balance = _as_float(self.snapshot.get("balance"), 0.0)
        capital = balance * self.settings.max_trade_balance_fraction * max(size_multiplier, 0.1)
        capital = max(capital, self.settings.min_trade_usdt)
        qty = self.order_manager.normalize_quantity(capital / max(price, 1e-9))
        return float(qty)

    def _build_stops(self, side: str, entry: float, atr: float) -> tuple[float, float]:
        atr = max(atr, entry * 0.002)
        if side == "BUY":
            stop_loss = self.order_manager.normalize_price(entry - (1.5 * atr))
            take_profit = self.order_manager.normalize_price(entry + (2.0 * atr))
        else:
            stop_loss = self.order_manager.normalize_price(entry + (1.5 * atr))
            take_profit = self.order_manager.normalize_price(entry - (2.0 * atr))
        return stop_loss, take_profit

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
            self.symbol,
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

    def _update_snapshot(self, market_data: dict[str, Any], analysis: dict[str, Any]) -> None:
        bundle: SignalBundle = analysis["bundle"]
        self.snapshot.update(
            {
                "pair": self.symbol,
                "timeframe": f"{self.settings.primary_timeframe} / {self.settings.confirmation_timeframe}",
                "price": market_data.get("price"),
                "spread": market_data.get("spread"),
                "trend": bundle.trend,
                "adx": market_data.get("adx"),
                "volatility_regime": bundle.regime,
                "order_flow": bundle.order_flow,
                "volume": market_data.get("volume"),
                "change_24h": market_data.get("change_24h"),
                "signal": analysis.get("side"),
                "confidence": analysis.get("confidence"),
                "status": self.state.value,
                "signals": {
                    "trend": bundle.trend,
                    "momentum": bundle.momentum,
                    "volume": bundle.volume,
                    "volatility": bundle.volatility,
                    "structure": bundle.structure,
                    "order_flow": bundle.order_flow,
                },
            }
        )

    async def _fetch_usdt_balance(self) -> float:
        payload = await self.client.fetch_balance()
        usdt = payload.get("USDT") or {}
        return _as_float(usdt.get("free"), 0.0)

    async def _set_state(self, new_state: BotState, context: str = "") -> None:
        if new_state == self.state:
            return
        previous = self.state
        self.state = new_state
        self.snapshot["status"] = new_state.value
        await self.repository.record_state_change(previous.value, new_state.value, context)

    async def _log(self, level: str, message: str) -> None:
        getattr(self.logger, level.lower(), self.logger.info)(message)
        await self.repository.record_log(level, self.state.value, message)
        if self.state_manager:
            self.state_manager.add_log(level, message)

    def _roll_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.day_marker:
            self.day_marker = today
            self.trades_today = 0
            self.win_count = 0

    def _is_cooldown_complete(self) -> bool:
        if self.last_close_time is None:
            return True
        return datetime.now(timezone.utc) - self.last_close_time >= timedelta(minutes=self.settings.cooldown_minutes)

    def _safe_live_update(self, live: Live) -> None:
        try:
            live.update(self.dashboard.render(self.snapshot))
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("dashboard_render_error: %s", exc)

    def _sync_ui_state(self) -> None:
        if not self.state_manager:
            return
        try:
            self.state_manager.update(
                pair=self.snapshot.get("pair", ""),
                timeframe=self.snapshot.get("timeframe", ""),
                current_price=self.snapshot.get("price"),
                spread=self.snapshot.get("spread"),
                trend=self.snapshot.get("trend"),
                adx=self.snapshot.get("adx"),
                volatility_regime=self.snapshot.get("volatility_regime"),
                order_flow=self.snapshot.get("order_flow"),
                signal=self.snapshot.get("signal"),
                confidence=self.snapshot.get("confidence"),
                balance=self.snapshot.get("balance"),
                equity=self.snapshot.get("equity"),
                daily_pnl=self.snapshot.get("daily_pnl"),
                trades_today=self.snapshot.get("trades_today", 0),
                win_rate=self.snapshot.get("win_rate"),
                last_trade=self.snapshot.get("last_trade"),
                cooldown=self.snapshot.get("cooldown"),
                status=self.snapshot.get("status", "INITIALIZING"),
                bid=self.snapshot.get("price"),
                ask=self.snapshot.get("price"),
                volume=self.snapshot.get("volume"),
                atr=0.0,
                change_24h=self.snapshot.get("change_24h"),
                signals=self.snapshot.get("signals", {}),
                system={
                    "uptime_seconds": time.time() - self.start_time,
                    "api_latency_ms": _as_float(self.snapshot.get("api_latency_ms"), 0.0),
                    "database_status": self.snapshot.get("database_status", "UNKNOWN"),
                    "exchange_status": self.snapshot.get("exchange_status", "UNKNOWN"),
                    "redis_status": self.snapshot.get("redis_status", "UNKNOWN"),
                    "memory_usage_mb": 0.0,
                    "last_server_sync": datetime.utcnow().isoformat(timespec="seconds"),
                },
                risk_metrics={
                    "risk_per_trade": f"{self.settings.risk_per_trade_fraction:.2%}",
                    "max_concurrent_trades": self.settings.max_trades_per_day,
                    "daily_drawdown": f"{max(0.0, -_as_float(self.snapshot.get('daily_pnl'), 0.0)):.4f}",
                    "consecutive_losses": self.consecutive_losses,
                    "current_exposure": f"{(self.settings.max_trade_balance_fraction * 100):.1f}%",
                },
                analytics={
                    "total_trades": self.trades_today,
                    "win_rate": self.snapshot.get("win_rate") or 0.0,
                    "profit_factor": 0.0,
                    "average_win": 0.0,
                    "average_loss": 0.0,
                    "largest_win": 0.0,
                    "largest_loss": 0.0,
                    "equity_curve": [self.snapshot.get("equity") or 0.0],
                },
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("state_sync_error: %s", exc)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _book_price(order_book: dict[str, Any], side: str, fallback: float) -> float:
    levels = order_book.get(side) or []
    if levels and levels[0]:
        return _as_float(levels[0][0], fallback)
    return fallback
