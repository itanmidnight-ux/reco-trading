from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
import time

import ccxt

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

    def __init__(self, settings: Settings, state_manager: "StateManager | None" = None) -> None:
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
        self.state_manager = state_manager
        self.start_time = time.time()

    async def run(self) -> None:
        await self.repository.setup()
        await self.client.sync_time()
        await self.order_manager.sync_rules()
        with Live(TerminalDashboard().render(self.snapshot), refresh_per_second=2, transient=False) as live:
            while True:
                try:
                    self._roll_day()
                    market_data = await self.fetch_market_data()
                    analysis = await self.analyze_market(market_data)

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

    async def _tick(self) -> None:
        self._roll_day()
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

        if self.signal_engine.is_sideways(frame5):
            await self._set_state(BotState.WAITING_SIGNAL, "sideways_market")
            await self._log("INFO", "sideways_market_detected")
            await self._refresh(price=price, bid=bid, ask=ask, spread=spread, atr=float(last_candle["atr"]))
            return

    async def analyze_market(self, market_data: dict[str, Any]) -> dict[str, Any]:
        frame5: pd.DataFrame = market_data["frame5"]
        frame15: pd.DataFrame = market_data["frame15"]
        bundle = self.signal_engine.generate(frame5, frame15)
        side, confidence, _grade = self.confidence_model.evaluate(bundle)
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

        await self._set_state(BotState.CHECKING_RISK)
        risk = self.risk_manager.validate(
            balance=balance,
            daily_pnl=daily_pnl,
            trades_today=self.trades_today,
            confidence=confidence,
            confidence_threshold=self.settings.confidence_threshold,
        )
        cooldown_ok = self._is_cooldown_complete()

        self._push_state(
            pair=self.settings.symbol,
            timeframe=f"{self.settings.primary_timeframe} / {self.settings.confirmation_timeframe}",
            current_price=price,
            bid=bid,
            ask=ask,
            spread=spread,
            trend=bundle.trend,
            signal=side,
            confidence=confidence,
            volatility_regime=bundle.regime,
            order_flow=bundle.order_flow,
            adx=float(last_candle.get("adx", 0.0)),
            atr=float(last_candle.get("atr", 0.0)),
            volume=float(last_candle.get("volume", 0.0)),
            balance=balance,
            equity=balance + daily_pnl,
            daily_pnl=daily_pnl,
            trades_today=self.trades_today,
            cooldown="READY" if cooldown_ok else "ACTIVE",
            signal_analysis={
                "trend": bundle.trend,
                "momentum": bundle.momentum,
                "volume_signal": bundle.volume,
                "order_flow": bundle.order_flow,
                "confidence": confidence,
            },
            risk_metrics={
                "risk_per_trade": f"{self.settings.max_trade_balance_fraction:.2%}",
                "max_trades_per_hour": self.settings.max_trades_per_day,
                "cooldown": self.settings.cooldown_minutes,
                "consecutive_losses": 0,
                "current_drawdown": f"{max(0.0, -daily_pnl):.4f}",
                "daily_exposure": f"{(self.settings.max_trade_balance_fraction * 100):.1f}%",
            },
            system={
                "uptime_seconds": time.time() - self.start_time,
                "api_latency_ms": 0.0,
                "database_status": "CONNECTED",
                "last_server_sync": datetime.utcnow().isoformat(),
            },
        )

        if not risk.approved or not bundle.regime_trade_allowed or not cooldown_ok:
            await self._set_state(BotState.PAUSED if risk.reason == "RISK_PAUSE" else BotState.WAITING_SIGNAL, risk.reason)
            await self._monitor_positions(price)
            await self._refresh(price=price, bid=bid, ask=ask, spread=spread, atr=float(last_candle["atr"]))
            return

        if self.position_manager.can_open(confidence):
            await self._open_trade(side, price, float(last_candle["atr"]), bundle.size_multiplier)

        await self._monitor_positions(price)
        await self._refresh(price=price, bid=bid, ask=ask, spread=spread, atr=float(last_candle["atr"]))

    async def _open_trade(self, side: str, price: float, atr: float, size_multiplier: float) -> None:
        await self._set_state(BotState.PLACING_ORDER)
        base_capital = (await self._fetch_usdt_balance()) * self.settings.max_trade_balance_fraction * size_multiplier
        qty = self.order_manager.normalize_quantity(base_capital / price)
        if qty <= 0:
            await self._log("WARNING", "quantity_below_minimum")
            return
        if not self.order_manager.validate_notional(qty, price):
            await self._log("WARNING", "notional_below_minimum")
            return

        try:
            order = await self.client.create_market_order(self.settings.symbol, side.lower(), qty)
        except ccxt.BaseError as exc:
            await self._log("ERROR", f"order_rejected error={exc}")
            await self.repository.record_error(self.state.value, "order", str(exc))
            return

        entry = float(order.get("average") or order.get("price") or price)
        stop_loss = self.order_manager.normalize_price(entry - (1.5 * atr))
        take_profit = self.order_manager.normalize_price(entry + (2.0 * atr))

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
        signal_analysis = self.state_manager.snapshot().get("signal_analysis", {}) if self.state_manager else {}
        trade_payload = {
            "trade_id": trade.id,
            "time": datetime.utcnow().isoformat(timespec="seconds"),
            "pair": self.settings.symbol,
            "side": side,
            "entry": entry,
            "size": qty,
            "position_value": qty * entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "OPEN",
            **signal_analysis,
        }
        if self.state_manager:
            self.state_manager.add_trade(trade_payload)
            self.state_manager.notify("Trade executed", f"{side} {self.settings.symbol} @ {entry:.2f}")
        await self._set_state(BotState.ORDER_FILLED)
        await self._log("INFO", f"order_filled side={side} qty={qty:.8f} entry={entry:.2f}")

    async def manage_open_position(self, market_data: dict[str, Any]) -> None:
        if not self.position_manager.positions:
            return
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
            await self._set_state(BotState.TAKE_PROFIT_HIT if "TAKE_PROFIT" in exit_reason else BotState.STOP_LOSS_HIT)
            if self.state_manager:
                self.state_manager.notify("Trade closed", f"id={position.trade_id} pnl={pnl:.4f}")
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
        self._push_state(status=new_state.value)

    async def _log(self, level: str, message: str) -> None:
        getattr(self.logger, level.lower(), self.logger.info)(message)
        await self.repository.record_log(level, self.state.value, message)
        if self.state_manager:
            self.state_manager.add_log(level, message)

    def _push_state(self, **payload: object) -> None:
        if self.state_manager:
            self.state_manager.update(**payload)

    def _roll_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.day_marker:
            self.day_marker = today
            self.trades_today = 0

    def _is_cooldown_complete(self) -> bool:
        if self.last_close_time is None:
            return True
        return datetime.now(timezone.utc) - self.last_close_time >= timedelta(minutes=self.settings.cooldown_minutes)

    async def _refresh(self, price: float, bid: float, ask: float, spread: float, atr: float) -> None:
        if self.position_manager.positions:
            pos = self.position_manager.positions[0]
            self._push_state(open_position=pos.side, last_trade=f"{pos.side} @ {pos.entry_price:.2f}")
        else:
            self._push_state(open_position="NONE")
        self._push_state(current_price=price, bid=bid, ask=ask, spread=spread, atr=atr)
        await asyncio.sleep(self.settings.loop_sleep_seconds)
