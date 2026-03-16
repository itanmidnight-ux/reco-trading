from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import ccxt
from rich.live import Live

from reco_trading.config.settings import Settings
from reco_trading.config.symbols import normalize_symbol, split_symbol
from reco_trading.core.state_machine import BotState
from reco_trading.data.market_stream import MarketStream
from reco_trading.database.repository import Repository
from reco_trading.exchange.binance_client import BinanceClient
from reco_trading.exchange.order_manager import OrderManager
from reco_trading.risk.position_manager import Position, PositionManager
from reco_trading.risk.advanced_risk_manager import AdvancedRiskManager
from reco_trading.risk.risk_manager import RiskManager
from reco_trading.strategy.confidence_model import ConfidenceModel
from reco_trading.strategy.indicators import apply_indicators
from reco_trading.strategy.market_intelligence import MarketIntelligence
from reco_trading.strategy.signal_engine import SignalBundle, SignalEngine
from reco_trading.ui.dashboard import TerminalDashboard

if TYPE_CHECKING:
    from reco_trading.ui.state_manager import StateManager


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
        self.advanced_risk_manager = AdvancedRiskManager(
            max_daily_loss_percent=max(float(settings.daily_loss_limit_fraction), 0.0) * 100,
            max_drawdown_percent=max(float(getattr(settings, "max_drawdown_fraction", 0.10)), 0.0) * 100,
        )
        self.position_manager = PositionManager()
        self.market_intelligence = MarketIntelligence(settings)

        self.state_manager = state_manager
        self.trades_today = 0
        self.win_count = 0
        self.start_time = time.time()
        self.day_marker = datetime.now(timezone.utc).date()
        self.last_close_time: datetime | None = None
        self.pause_trading_until: datetime | None = None
        self.consecutive_losses = 0
        self.equity_peak: float | None = None
        self.starting_equity: float | None = None
        self.trading_paused_by_drawdown = False
        self.exchange_failure_count = 0
        self.exchange_failure_max = 5
        self.exchange_failure_cooldown_seconds = 300
        self.exchange_failure_paused_until: datetime | None = None

        self._cached_frame5: Any | None = None
        self._cached_frame15: Any | None = None
        self._last_primary_indicator_ts: datetime | None = None
        self._last_confirmation_indicator_ts: datetime | None = None

        self.dashboard = TerminalDashboard()
        self.snapshot: dict[str, Any] = {
            "pair": self.symbol,
            "timeframe": f"{self.settings.primary_timeframe} / {self.settings.confirmation_timeframe}",
            "price": None,
            "spread": None,
            "bid": None,
            "ask": None,
            "trend": None,
            "adx": None,
            "volatility_regime": None,
            "order_flow": None,
            "signal": None,
            "confidence": None,
            "balance": None,
            "equity": None,
            "btc_balance": 0.0,
            "btc_value": 0.0,
            "total_equity": None,
            "daily_pnl": None,
            "session_pnl": None,
            "trades_today": 0,
            "win_rate": None,
            "last_trade": None,
            "cooldown": None,
            "status": BotState.INITIALIZING.value,
            "signals": {},
            "volume": None,
            "atr": None,
            "api_latency_ms": None,
            "candles_5m": [],
            "started_at": time.time(),
            "market_regime": None,
            "volatility_state": None,
            "distance_to_support": None,
            "distance_to_resistance": None,
        }

    async def run(self) -> None:
        try:
            await self._set_state(BotState.INITIALIZING, "initialize_settings")
            await self.repository.setup()
            await self._set_state(BotState.CONNECTING_EXCHANGE, "connect_exchange")
            await self.client.sync_time()
            await self._set_state(BotState.SYNCING_SYMBOL, "sync_symbol")
            await self._set_state(BotState.SYNCING_RULES, "sync_exchange_rules")
            await self.order_manager.sync_rules()
            await self._reconcile_open_positions()
            await self._set_state(BotState.WAITING_MARKET_DATA, "ready")
            self._sync_ui_state()

            with Live(self.dashboard.render(self.snapshot), refresh_per_second=2, transient=False) as live:
                while True:
                    try:
                        if self.exchange_failure_paused_until and datetime.now(timezone.utc) < self.exchange_failure_paused_until:
                            await self._set_state(BotState.PAUSED, "exchange_circuit_breaker")
                            self.snapshot["cooldown"] = f"EXCHANGE_PAUSED until {self.exchange_failure_paused_until.isoformat(timespec='seconds')}"
                            self._sync_ui_state()
                            self._safe_live_update(live)
                            await asyncio.sleep(self.settings.loop_sleep_seconds)
                            continue

                        self._roll_day()
                        await self._set_state(BotState.WAITING_MARKET_DATA, "fetch_market_data")
                        market_data = await self.fetch_market_data()
                        if not self._is_market_data_fresh(market_data):
                            await self._set_state(BotState.WAITING_MARKET_DATA, "stale_market_data")
                            self.snapshot["cooldown"] = "STALE_MARKET_DATA"
                            self._sync_ui_state()
                            self._safe_live_update(live)
                            await asyncio.sleep(self.settings.loop_sleep_seconds)
                            continue
                        await self._process_control_requests()
                        await self._set_state(BotState.ANALYZING_MARKET, "analyze_market")
                        analysis = await self.analyze_market(market_data)
                        self._update_snapshot(market_data, analysis)

                        if await self.validate_trade_conditions(analysis):
                            intelligence = self.market_intelligence.evaluate(str(analysis.get("side", "HOLD")), market_data)
                            self._apply_market_intelligence_snapshot(intelligence)
                            if intelligence.get("approved"):
                                await self.execute_trade(analysis, market_data, float(intelligence.get("size_multiplier", 1.0)))
                            else:
                                await self._set_state(BotState.WAITING_MARKET_DATA, str(intelligence.get("reason", "market_intelligence")))

                        await self.manage_open_position(market_data)
                        self.exchange_failure_count = 0
                        self._sync_ui_state()
                        self._safe_live_update(live)
                        await asyncio.sleep(self.settings.loop_sleep_seconds)
                    except KeyboardInterrupt:
                        await self._set_state(BotState.STOPPED, "manual_stop")
                        break
                    except ccxt.BaseError as exc:
                        await self._set_state(BotState.ERROR, "exchange_error")
                        self._register_exchange_failure()
                        await self._log("ERROR", f"exchange_error={exc}")
                        await self.repository.record_error(self.state.value, "exchange", str(exc))
                        self.snapshot["status"] = BotState.ERROR.value
                        self._sync_ui_state()
                        self._safe_live_update(live)
                        await asyncio.sleep(self.settings.loop_sleep_seconds)
                    except Exception as exc:  # noqa: BLE001
                        await self._set_state(BotState.ERROR, "runtime_error")
                        await self._log("ERROR", f"runtime_error={exc}")
                        await self.repository.record_error(self.state.value, "runtime", str(exc))
                        self.snapshot["status"] = BotState.ERROR.value
                        self._safe_live_update(live)
                        await asyncio.sleep(self.settings.loop_sleep_seconds)
        finally:
            await self.client.close()
            await self.repository.close()

    async def fetch_market_data(self) -> dict[str, Any]:
        raw_frame5, raw_frame15 = await asyncio.gather(
            self.market_stream.fetch_frame(self.settings.primary_timeframe),
            self.market_stream.fetch_frame(self.settings.confirmation_timeframe),
        )

        latest_primary_ts = _timestamp_to_datetime(raw_frame5.iloc[-1].get("timestamp") if not raw_frame5.empty else None)
        latest_confirmation_ts = _timestamp_to_datetime(raw_frame15.iloc[-1].get("timestamp") if not raw_frame15.empty else None)

        if (
            self._cached_frame5 is not None
            and latest_primary_ts is not None
            and latest_primary_ts == self._last_primary_indicator_ts
        ):
            frame5 = self._cached_frame5
        else:
            frame5 = apply_indicators(raw_frame5)
            self._cached_frame5 = frame5
            self._last_primary_indicator_ts = latest_primary_ts

        if (
            self._cached_frame15 is not None
            and latest_confirmation_ts is not None
            and latest_confirmation_ts == self._last_confirmation_indicator_ts
        ):
            frame15 = self._cached_frame15
        else:
            frame15 = apply_indicators(raw_frame15)
            self._cached_frame15 = frame15
            self._last_confirmation_indicator_ts = latest_confirmation_ts

        candle = frame5.iloc[-1]
        candles_5m = self._frame_to_candles(frame5)

        tick_start = time.perf_counter()
        ticker, order_book = await asyncio.gather(
            self.client.fetch_ticker(self.symbol),
            self.client.fetch_order_book(self.symbol),
        )
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
            "candles_5m": candles_5m,
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
        await self._set_state(BotState.SIGNAL_GENERATED, "analysis_complete")
        return {"bundle": bundle, "side": side, "confidence": confidence, "grade": grade}

    async def validate_trade_conditions(self, analysis: dict[str, Any]) -> bool:
        confidence = float(analysis["confidence"])
        side = str(analysis.get("side", "HOLD")).upper()
        if side == "HOLD":
            await self._set_state(BotState.WAITING_MARKET_DATA, "hold_signal")
            self.snapshot["cooldown"] = "HOLD_SIGNAL"
            return False

        if getattr(self.settings, "spot_only_mode", True) and side == "SELL" and not self.position_manager.positions:
            await self._set_state(BotState.WAITING_MARKET_DATA, "spot_short_blocked")
            self.snapshot["cooldown"] = "SPOT_SHORT_BLOCKED"
            return False

        usdt_balance, btc_balance = await self._fetch_balances()
        current_price = _as_float(self.snapshot.get("price"), 0.0)
        btc_value = float(btc_balance * current_price)
        total_equity = float(usdt_balance + btc_value)
        session_pnl = float(await self.repository.get_session_pnl() or 0.0)

        self.snapshot["balance"] = float(usdt_balance)
        self.snapshot["btc_balance"] = float(btc_balance)
        self.snapshot["btc_value"] = btc_value
        self.snapshot["total_equity"] = total_equity
        self.snapshot["equity"] = total_equity
        self.snapshot["daily_pnl"] = session_pnl
        self.snapshot["session_pnl"] = session_pnl
        self.snapshot["trades_today"] = self.trades_today

        self.equity_peak = max(_as_float(self.equity_peak, total_equity), total_equity)
        if self.starting_equity is None:
            self.starting_equity = max(total_equity, 1.0)
        max_drawdown_fraction = _as_float(getattr(self.settings, "max_drawdown_fraction", 0.10), 0.10)
        if self.equity_peak > 0:
            drawdown = max((self.equity_peak - total_equity) / self.equity_peak, 0.0)
            if drawdown >= max_drawdown_fraction:
                self.trading_paused_by_drawdown = True

        if self.trading_paused_by_drawdown:
            await self._set_state(BotState.PAUSED, "max_drawdown")
            self.snapshot["cooldown"] = "MAX_DRAWDOWN"
            return False

        if self.pause_trading_until and datetime.now(timezone.utc) < self.pause_trading_until:
            await self._set_state(BotState.PAUSED, "loss_protection_pause")
            self.snapshot["cooldown"] = f"PAUSED until {self.pause_trading_until.isoformat(timespec='seconds')}"
            return False

        if not self._is_cooldown_complete():
            await self._set_state(BotState.COOLDOWN, "cooldown_active")
            self.snapshot["cooldown"] = "ACTIVE"
            return False

        risk = self.risk_manager.validate(
            balance=usdt_balance,
            daily_pnl=session_pnl,
            trades_today=self.trades_today,
            confidence=confidence,
            confidence_threshold=self.settings.confidence_threshold,
        )
        if not risk.approved:
            await self._set_state(BotState.PAUSED if risk.reason == "RISK_PAUSE" else BotState.WAITING_MARKET_DATA, risk.reason)
            self.snapshot["cooldown"] = risk.reason
            return False

        volatility_ratio = _as_float(self.snapshot.get("atr"), 0.0) / max(current_price, 1e-9)
        advanced = self.advanced_risk_manager.evaluate(
            daily_pnl=session_pnl,
            starting_equity=max(_as_float(self.starting_equity, total_equity), 1.0),
            consecutive_losses=self.consecutive_losses,
            current_equity=total_equity,
            peak_equity=max(_as_float(self.equity_peak, total_equity), 1.0),
            volatility_ratio=volatility_ratio,
        )
        if not advanced.approved:
            await self._set_state(BotState.PAUSED, advanced.reason)
            self.snapshot["cooldown"] = advanced.reason
            if advanced.pause_trading:
                self.pause_trading_until = datetime.now(timezone.utc) + timedelta(minutes=self.settings.loss_pause_minutes)
            return False

        self.snapshot["advanced_risk_reason"] = advanced.reason
        self.snapshot["advanced_size_multiplier"] = advanced.size_multiplier

        self.snapshot["cooldown"] = "READY"
        return True

    async def execute_trade(self, analysis: dict[str, Any], market_data: dict[str, Any], intelligence_size_multiplier: float = 1.0) -> None:
        bundle: SignalBundle = analysis["bundle"]
        side = str(analysis["side"]).upper()
        price = float(market_data["price"])
        atr = float(market_data.get("atr", 0.0))

        if side not in {"BUY", "SELL"}:
            await self._set_state(BotState.WAITING_MARKET_DATA, "invalid_side")
            return

        if getattr(self.settings, "spot_only_mode", True) and side != "BUY":
            await self._set_state(BotState.WAITING_MARKET_DATA, "spot_open_only_buy")
            return

        if not bundle.regime_trade_allowed:
            await self._set_state(BotState.WAITING_MARKET_DATA, "regime_filter")
            return

        if not self.position_manager.can_open(self.settings.max_concurrent_trades):
            await self._set_state(BotState.POSITION_OPEN, "max_positions")
            return

        spread = _as_float(market_data.get("spread"), 0.0)
        spread_ratio = spread / max(price, 1e-9)
        if spread_ratio > _as_float(self.settings.max_spread_ratio, 0.002):
            await self._set_state(BotState.WAITING_MARKET_DATA, "spread_too_wide")
            return

        if not self._pullback_confirmed(bundle, side, market_data):
            await self._set_state(BotState.WAITING_MARKET_DATA, "pullback_unconfirmed")
            return

        stop_loss, take_profit = self._build_stops(side, price, atr)
        qty = self.calculate_position_size(
            price,
            stop_loss,
            atr,
            float(bundle.size_multiplier)
            * max(float(intelligence_size_multiplier), 0.1)
            * max(_as_float(self.snapshot.get("advanced_size_multiplier"), 1.0), 0.1),
        )
        if qty <= 0:
            await self._log("WARNING", "quantity_below_minimum")
            return

        original_qty = qty
        equity = max(_as_float(self.snapshot.get("equity"), _as_float(self.snapshot.get("balance"), 0.0)), 0.0)
        normalized_qty = self.order_manager.normalize_order_quantity(
            symbol=self.symbol,
            price=price,
            quantity=qty,
            equity=equity,
            max_trade_balance_fraction=float(self.settings.max_trade_balance_fraction),
        )
        if normalized_qty is None:
            await self._log(
                "WARNING",
                f"order_rejected_after_normalization symbol={self.symbol} original_quantity={original_qty:.8f} price={price:.8f}",
            )
            return
        qty = normalized_qty
        if abs(qty - original_qty) > 0:
            rules = self.order_manager.rules
            notional = qty * price
            await self._log(
                "INFO",
                "Adjusted order quantity to satisfy Binance filters "
                f"symbol={self.symbol} original_quantity={original_qty:.8f} normalized_quantity={qty:.8f} "
                f"price={price:.8f} notional={notional:.8f} minQty={rules.min_qty:.8f} "
                f"stepSize={rules.step_size:.8f} minNotional={rules.min_notional:.8f}",
            )

        if not self.order_manager.validate_notional(qty, price):
            await self._log("WARNING", "notional_below_minimum")
            return

        await self._set_state(BotState.PLACING_ORDER)
        try:
            intent_id = f"reco-open-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
            order = await self.client.create_market_order(self.symbol, side.lower(), qty, client_order_id=intent_id)
        except ccxt.BaseError as exc:
            await self._log("ERROR", f"order_rejected error={exc}")
            await self.repository.record_error(self.state.value, "order", str(exc))
            return

        entry = _as_float(order.get("average"), _as_float(order.get("price"), price))
        slippage_ratio = abs(entry - price) / max(price, 1e-9)
        if slippage_ratio > _as_float(self.settings.max_slippage_ratio, 0.003):
            await self._log("ERROR", f"slippage_too_high slippage_ratio={slippage_ratio:.6f}")
            try:
                close_side = "sell" if side == "BUY" else "buy"
                close_intent = f"reco-close-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
                await self.client.create_market_order(self.symbol, close_side, qty, client_order_id=close_intent)
            except ccxt.BaseError as exc:
                await self.repository.record_error(self.state.value, "slippage_exit", str(exc))
            self.pause_trading_until = datetime.now(timezone.utc) + timedelta(minutes=15)
            return

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
                initial_risk_distance=abs(entry - stop_loss),
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
            await self._log("INFO", f"exit_condition_detected trade_id={position.trade_id} reason={exit_reason} market_price={price:.8f}")
            closed = await self._close_position(position, exit_reason, price)
            if not closed:
                await self._log("ERROR", f"exit_close_failed trade_id={position.trade_id} reason={exit_reason} action=retry_next_loop")

    async def force_close_position(self) -> None:
        if not self.position_manager.positions:
            await self._log("INFO", "MANUAL_TRADE_CLOSE_SKIPPED no_open_position")
            return

        await self._set_state(BotState.COOLDOWN)
        for position in list(self.position_manager.positions):
            await self._log("INFO", f"MANUAL_TRADE_CLOSE requested trade_id={position.trade_id}")
            closed = await self._close_position(position, "MANUAL_TRADE_CLOSE", _as_float(self.snapshot.get("price"), position.entry_price))
            if not closed:
                await self._log("ERROR", f"MANUAL_TRADE_CLOSE_FAILED trade_id={position.trade_id} action=retry_next_loop")

    async def _close_position(self, position: Position, exit_reason: str, reference_price: float) -> bool:
        close_side = "sell" if position.side == "BUY" else "buy"
        order = None
        for attempt in range(1, 4):
            try:
                intent_id = f"reco-close-{position.trade_id}-{attempt}-{uuid.uuid4().hex[:6]}"
                order = await self.client.create_market_order(
                    self.symbol,
                    close_side,
                    position.quantity,
                    client_order_id=intent_id,
                )
                break
            except ccxt.BaseError as exc:
                await self._log(
                    "WARNING",
                    f"close_order_attempt_failed trade_id={position.trade_id} reason={exit_reason} attempt={attempt}/3 error={exc}",
                )
                if attempt < 3:
                    await asyncio.sleep(1)

        if order is None:
            await self.repository.record_error(self.state.value, "close_order", f"trade_id={position.trade_id} reason={exit_reason}")
            return False

        exit_price = _as_float(order.get("average"), _as_float(order.get("price"), reference_price))
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

        await self._log("INFO", f"position_closed trade_id={position.trade_id} reason={exit_reason} exit_price={exit_price:.8f} pnl={pnl:.8f}")
        self._sync_ui_state()

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
        return True

    def calculate_position_size(self, price: float, stop_loss: float, atr: float, size_multiplier: float) -> float:
        equity = _as_float(self.snapshot.get("equity"), _as_float(self.snapshot.get("balance"), 0.0))
        sizing = self.risk_manager.position_size_for_risk(
            equity=equity,
            risk_fraction=self.settings.risk_per_trade_fraction,
            price=price,
            stop_loss_price=stop_loss,
            atr=atr,
            atr_floor_multiplier=0.5,
        )
        # Keep raw risk-derived size here. Binance filter normalization (minQty/stepSize/minNotional)
        # is applied centrally in execute_trade via normalize_order_quantity.
        qty = sizing.quantity * max(size_multiplier, 0.1)
        return float(max(qty, 0.0))

    def _pullback_confirmed(self, bundle: SignalBundle, side: str, market_data: dict[str, Any]) -> bool:
        frame5 = market_data.get("frame5")
        if frame5 is None or len(frame5) < 5:
            return False
        row = frame5.iloc[-1]
        prev = frame5.iloc[-2]
        close = _as_float(row.get("close"), _as_float(market_data.get("price"), 0.0))
        open_px = _as_float(row.get("open"), close)
        ema20 = _as_float(row.get("ema20"), close)
        recent_high = _as_float(frame5["high"].tail(5).max(), close)
        recent_low = _as_float(frame5["low"].tail(5).min(), close)

        if side == "BUY":
            pullback = close <= ema20 or close < (recent_high * 0.999)
            reversal = close > open_px and close > _as_float(prev.get("close"), close)
            return pullback and reversal

        pullback = close >= ema20 or close > (recent_low * 1.001)
        rejection = close < open_px and close < _as_float(prev.get("close"), close)
        return pullback and rejection

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
                "bid": market_data.get("bid"),
                "ask": market_data.get("ask"),
                "trend": bundle.trend,
                "adx": market_data.get("adx"),
                "volatility_regime": bundle.regime,
                "order_flow": bundle.order_flow,
                "volume": market_data.get("volume"),
                "atr": market_data.get("atr"),
                "change_24h": market_data.get("change_24h"),
                "candles_5m": list(market_data.get("candles_5m") or [])[-120:],
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


    def _apply_market_intelligence_snapshot(self, intelligence: dict[str, Any]) -> None:
        self.snapshot["market_regime"] = intelligence.get("market_regime")
        self.snapshot["volatility_state"] = intelligence.get("volatility_state")
        self.snapshot["distance_to_support"] = intelligence.get("distance_to_support")
        self.snapshot["distance_to_resistance"] = intelligence.get("distance_to_resistance")

    async def _fetch_balances(self) -> tuple[float, float]:
        payload = await self.client.fetch_balance()
        base_asset, quote_asset = split_symbol(self.symbol)
        quote_asset = quote_asset or "USDT"

        total_balances = payload.get("total") if isinstance(payload, dict) else None
        if isinstance(total_balances, dict):
            quote_balance = _as_float(total_balances.get(quote_asset), 0.0)
            base_balance = _as_float(total_balances.get(base_asset), 0.0)
            return quote_balance, base_balance

        quote = payload.get(quote_asset) if isinstance(payload, dict) else {}
        base = payload.get(base_asset) if isinstance(payload, dict) else {}
        quote_balance = _as_float((quote or {}).get("free"), 0.0)
        base_balance = _as_float((base or {}).get("free"), 0.0)
        return quote_balance, base_balance

    async def _reconcile_open_positions(self) -> None:
        open_trades = await self.repository.get_open_trades(self.symbol)
        if not open_trades:
            return
        usdt_balance, base_balance = await self._fetch_balances()
        self.snapshot["balance"] = usdt_balance
        self.snapshot["btc_balance"] = base_balance
        if base_balance <= 0:
            await self._log("WARNING", "open_trade_found_without_base_balance")
            return
        remaining_base = base_balance
        reconciled = 0
        for trade in open_trades:
            if remaining_base <= 0:
                break
            qty = min(_as_float(trade.quantity, 0.0), remaining_base)
            if qty <= 0:
                continue
            self.position_manager.open(
                Position(
                    trade_id=trade.id,
                    side=trade.side,
                    quantity=qty,
                    entry_price=trade.entry_price,
                    stop_loss=trade.stop_loss,
                    take_profit=trade.take_profit,
                    atr=0.0,
                    initial_risk_distance=abs(trade.entry_price - trade.stop_loss),
                )
            )
            remaining_base -= qty
            reconciled += 1
        await self._log("INFO", f"reconciled_open_positions count={reconciled} base_balance={base_balance:.8f}")

    async def _process_control_requests(self) -> None:
        if not self.state_manager or not hasattr(self.state_manager, "pop_control_requests"):
            return
        controls = self.state_manager.pop_control_requests()
        for control in controls:
            if control == "force_close":
                await self.force_close_position()

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
                btc_balance=self.snapshot.get("btc_balance", 0.0),
                btc_value=self.snapshot.get("btc_value", 0.0),
                total_equity=self.snapshot.get("total_equity", self.snapshot.get("equity")),
                daily_pnl=self.snapshot.get("daily_pnl"),
                session_pnl=self.snapshot.get("session_pnl"),
                trades_today=self.snapshot.get("trades_today", 0),
                win_rate=self.snapshot.get("win_rate"),
                last_trade=self.snapshot.get("last_trade"),
                has_open_position=bool(self.position_manager.positions),
                cooldown=self.snapshot.get("cooldown"),
                status=self.snapshot.get("status", "INITIALIZING"),
                bid=self.snapshot.get("bid", self.snapshot.get("price")),
                ask=self.snapshot.get("ask", self.snapshot.get("price")),
                volume=self.snapshot.get("volume"),
                market_regime=self.snapshot.get("market_regime"),
                volatility_state=self.snapshot.get("volatility_state"),
                distance_to_support=self.snapshot.get("distance_to_support"),
                distance_to_resistance=self.snapshot.get("distance_to_resistance"),
                atr=self.snapshot.get("atr", 0.0),
                change_24h=self.snapshot.get("change_24h"),
                signals=self.snapshot.get("signals", {}),
                candles_5m=list(self.snapshot.get("candles_5m") or [])[-120:],
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
                    "max_concurrent_trades": self.settings.max_concurrent_trades,
                    "daily_drawdown": f"{max(0.0, -_as_float(self.snapshot.get('daily_pnl'), 0.0)):.4f}",
                    "consecutive_losses": self.consecutive_losses,
                    "current_exposure": self._current_exposure(),
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

    def _current_exposure(self) -> float:
        equity = _as_float(self.snapshot.get("equity"), 0.0)
        if equity <= 0:
            return 0.0
        mark_price = _as_float(self.snapshot.get("price"), 0.0)
        position_value = sum(max(_as_float(p.quantity, 0.0) * mark_price, 0.0) for p in self.position_manager.positions)
        return position_value / equity

    def _is_market_data_fresh(self, market_data: dict[str, Any]) -> bool:
        candle = market_data.get("candle")
        candle_timestamp = _timestamp_to_datetime(candle.get("timestamp") if candle is not None else None)
        if candle_timestamp is None:
            return False
        max_age_seconds = max(_timeframe_to_seconds(self.settings.primary_timeframe) * 2, 1)
        age_seconds = (datetime.now(timezone.utc) - candle_timestamp).total_seconds()
        return age_seconds <= max_age_seconds

    def _register_exchange_failure(self) -> None:
        self.exchange_failure_count += 1
        if self.exchange_failure_count < self.exchange_failure_max:
            return
        self.exchange_failure_paused_until = datetime.now(timezone.utc) + timedelta(seconds=self.exchange_failure_cooldown_seconds)
        self.exchange_failure_count = 0
        self.snapshot["cooldown"] = "EXCHANGE_CIRCUIT_BREAKER"
        self.logger.critical(
            "exchange_circuit_breaker_triggered pause_until=%s",
            self.exchange_failure_paused_until.isoformat(timespec="seconds"),
        )

    def _frame_to_candles(self, frame: Any) -> list[dict[str, float]]:
        candles: list[dict[str, float]] = []
        try:
            frame_slice = frame.tail(120)
        except Exception:  # noqa: BLE001
            return candles

        for _, row in frame_slice.iterrows():
            try:
                candles.append(
                    {
                        "open": _as_float(row.get("open"), 0.0),
                        "high": _as_float(row.get("high"), 0.0),
                        "low": _as_float(row.get("low"), 0.0),
                        "close": _as_float(row.get("close"), 0.0),
                        "volume": _as_float(row.get("volume"), 0.0),
                    }
                )
            except Exception:  # noqa: BLE001
                continue
        return candles


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


def _timeframe_to_seconds(timeframe: str) -> int:
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    return max(value, 1)


def _timestamp_to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None
    return parsed
