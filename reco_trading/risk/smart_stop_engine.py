from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from enum import Enum
import math

logger = logging.getLogger(__name__)


class StopType(Enum):
    FIXED = "fixed"
    TRAILING = "trailing"
    BREAK_EVEN = "break_even"
    DYNAMIC_ATR = "dynamic_atr"
    SMART_PROFIT = "smart_profit"
    EMERGENCY = "emergency"


@dataclass
class SmartStopConfig:
    enabled: bool = True
    initial_stop_atr_multiplier: float = 1.5
    trailing_activation_profit_r: float = 0.35
    trailing_atr_multiplier: float = 1.0
    break_even_trigger_profit_r: float = 0.25
    break_even_buffer_profit_r: float = 0.08
    profit_lock_trigger_r: float = 0.50
    profit_lock_buffer_ratio: float = 0.15
    time_decay_enabled: bool = True
    time_decay_start_minutes: int = 15
    time_decay_aggression: float = 0.3
    volatility_adjustment: bool = True
    max_stop_distance_atr: float = 3.0
    min_profit_to_trail_r: float = 0.20


@dataclass
class SmartStopDecision:
    should_update: bool
    new_stop_price: float | None
    stop_type: str
    reason: str
    urgency: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SmartStopEngine:
    """
    Motor inteligente de Stop Loss con multiples estrategias:
    - Break-even automático con buffer
    - Trailing stop dinámico basado en ATR
    - Profit locking progresivo
    - Decay temporal para evitar trades estancados
    - Ajuste por volatilidad
    - Protección de ganancias minimas
    """

    def __init__(self, config: SmartStopConfig | None = None):
        self.config = config or SmartStopConfig()
        self._position_peaks: dict[str, float] = {}
        self._position_entry_times: dict[str, datetime] = {}
        self._position_initial_stops: dict[str, float] = {}
        self._position_high_watermarks: dict[str, float] = {}
        self._position_current_stops: dict[str, float] = {}

    def initialize_position(
        self,
        position_id: str,
        entry_price: float,
        initial_stop: float,
        atr: float,
        side: str,
    ) -> None:
        """Inicializa el seguimiento de una posición."""
        self._position_peaks[position_id] = entry_price
        self._position_entry_times[position_id] = datetime.now(timezone.utc)
        self._position_initial_stops[position_id] = initial_stop
        self._position_high_watermarks[position_id] = entry_price
        self._position_current_stops[position_id] = initial_stop
        logger.debug(
            f"SmartStop initialized: pos={position_id} entry={entry_price} "
            f"stop={initial_stop} atr={atr:.4f}"
        )

    def evaluate(
        self,
        position_id: str,
        current_price: float,
        entry_price: float,
        atr: float,
        side: str,
        current_stop: float,
        equity: float,
        position_value: float,
        market_data: dict[str, Any] | None = None,
    ) -> SmartStopDecision:
        """
        Evalua y calcula el nuevo nivel de stop loss.
        
        Retorna decisión con nuevo stop price si debe actualizarse.
        """
        if not self.config.enabled:
            return SmartStopDecision(False, None, "disabled", "Smart stop disabled", 0.0)

        safe_atr = max(atr, entry_price * 0.001)
        safe_price = max(current_price, 1e-9)
        
        initial_risk_distance = abs(entry_price - current_stop)
        if initial_risk_distance < safe_atr * 0.1:
            initial_risk_distance = safe_atr * 1.5

        if side == "BUY":
            profit = safe_price - entry_price
            profit_r = profit / max(initial_risk_distance, 1e-9)
            peak = max(self._position_peaks.get(position_id, entry_price), safe_price)
            adverse_move = peak - safe_price
        else:
            profit = entry_price - safe_price
            profit_r = profit / max(initial_risk_distance, 1e-9)
            peak = min(self._position_peaks.get(position_id, entry_price), safe_price)
            adverse_move = safe_price - peak

        self._position_peaks[position_id] = peak
        self._position_high_watermarks[position_id] = max(
            self._position_high_watermarks.get(position_id, entry_price),
            peak if side == "BUY" else entry_price - (entry_price - peak)
        )

        giveback_r = adverse_move / max(initial_risk_distance, 1e-9)

        volatility_factor = self._calculate_volatility_factor(market_data, safe_atr, safe_price)
        time_factor = self._calculate_time_factor(position_id)
        urgency = self._calculate_urgency(profit_r, giveback_r, time_factor, volatility_factor)

        decision = self._apply_stop_strategy(
            position_id=position_id,
            side=side,
            entry_price=entry_price,
            current_price=safe_price,
            current_stop=current_stop,
            atr=safe_atr,
            profit_r=profit_r,
            giveback_r=giveback_r,
            peak=peak,
            volatility_factor=volatility_factor,
            time_factor=time_factor,
            urgency=urgency,
            equity=equity,
            position_value=position_value,
            market_data=market_data,
        )

        if decision.should_update and decision.new_stop_price:
            self._position_current_stops[position_id] = decision.new_stop_price

        return decision

    def _apply_stop_strategy(
        self,
        position_id: str,
        side: str,
        entry_price: float,
        current_price: float,
        current_stop: float,
        atr: float,
        profit_r: float,
        giveback_r: float,
        peak: float,
        volatility_factor: float,
        time_factor: float,
        urgency: float,
        equity: float,
        position_value: float,
        market_data: dict[str, Any] | None,
    ) -> SmartStopDecision:
        """
        Aplica la estrategia de stop más apropiada basada en:
        - Profit level (R multiple)
        - Volatility regime
        - Time in trade
        - Capital size
        - Giveback from peak
        """

        capital_ratio = min(position_value / max(equity, 1.0), 1.0)
        is_small_capital = equity < 50.0
        is_medium_capital = 50.0 <= equity < 500.0
        
        # More conservative for small capital
        if is_small_capital:
            conservative_multiplier = 0.6
        elif is_medium_capital:
            conservative_multiplier = 0.85
        else:
            conservative_multiplier = 1.0
        
        # Adjust thresholds based on volatility regime
        profit_lock_threshold = self.config.profit_lock_trigger_r
        trailing_threshold = self.config.trailing_activation_profit_r
        breakeven_threshold = self.config.break_even_trigger_profit_r
        
        # In high volatility, require more profit before locking/profit taking
        if volatility_factor > 1.1:
            profit_lock_threshold *= 1.15
            trailing_threshold *= 1.1
        # In low volatility, be more aggressive with locking profits
        elif volatility_factor < 0.95:
            profit_lock_threshold *= 0.9
            trailing_threshold *= 0.85
        
        # High giveback requires immediate action
        if giveback_r > 0.5:
            return self._emergency_protection_stop(
                side=side,
                entry_price=entry_price,
                current_price=current_price,
                atr=atr,
                giveback_r=giveback_r,
                profit_r=profit_r,
            )

        # Profit lock (highest priority - we have good profits)
        if profit_r >= profit_lock_threshold:
            return self._profit_lock_stop(
                side=side,
                entry_price=entry_price,
                current_price=current_price,
                atr=atr,
                profit_r=profit_r,
                giveback_r=giveback_r,
                peak=peak,
                capital_ratio=capital_ratio,
                conservative_multiplier=conservative_multiplier,
            )

        # Trailing stop (good profit, follow the trend)
        if profit_r >= trailing_threshold:
            return self._trailing_stop(
                side=side,
                entry_price=entry_price,
                current_price=current_price,
                current_stop=current_stop,
                atr=atr,
                profit_r=profit_r,
                giveback_r=giveback_r,
                peak=peak,
                volatility_factor=volatility_factor,
                conservative_multiplier=conservative_multiplier,
            )

        # Break-even stop (modest profit, protect capital)
        if profit_r >= breakeven_threshold:
            return self._break_even_stop(
                side=side,
                entry_price=entry_price,
                current_price=current_price,
                current_stop=current_stop,
                atr=atr,
                profit_r=profit_r,
                conservative_multiplier=conservative_multiplier,
            )

        # Time decay (position has been open too long without progress)
        if time_factor > 0.0:
            return self._time_decay_stop(
                side=side,
                entry_price=entry_price,
                current_price=current_price,
                current_stop=current_stop,
                atr=atr,
                time_factor=time_factor,
                profit_r=profit_r,
            )

        # Dynamic ATR stop (default)
        return self._dynamic_atr_stop(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            current_stop=current_stop,
            atr=atr,
            volatility_factor=volatility_factor,
            profit_r=profit_r,
        )

    def _emergency_protection_stop(
        self,
        side: str,
        entry_price: float,
        current_price: float,
        atr: float,
        giveback_r: float,
        profit_r: float,
    ) -> SmartStopDecision:
        """
        Emergency stop when price has given back too much from peak.
        Tightens stop aggressively to protect remaining profit.
        """
        # Lock in remaining profit minus a buffer
        remaining_profit_ratio = max(0.1, profit_r - giveback_r * 0.5)
        
        if side == "BUY":
            # Protect at least remaining_profit_ratio of the peak gain
            gain_from_entry = current_price - entry_price
            protected_gain = gain_from_entry * remaining_profit_ratio
            new_stop = entry_price + protected_gain
        else:
            gain_from_entry = entry_price - current_price
            protected_gain = gain_from_entry * remaining_profit_ratio
            new_stop = entry_price - protected_gain
        
        return SmartStopDecision(
            should_update=True,
            new_stop_price=round(new_stop, 8),
            stop_type=StopType.EMERGENCY.value,
            reason=f"EMERGENCY: High giveback ({giveback_r:.1%}), protecting remaining profit",
            urgency=1.0,
            metadata={
                "giveback_r": giveback_r,
                "profit_r": profit_r,
                "remaining_profit_ratio": remaining_profit_ratio,
            }
        )

    def _profit_lock_stop(
        self,
        side: str,
        entry_price: float,
        current_price: float,
        atr: float,
        profit_r: float,
        giveback_r: float,
        peak: float,
        capital_ratio: float,
        conservative_multiplier: float,
    ) -> SmartStopDecision:
        """Lock de ganancias cuando el trade va muy bien."""
        
        lock_buffer = self.config.profit_lock_buffer_ratio * profit_r
        
        if profit_r >= 1.0:
            lock_buffer = 0.08 * profit_r
        elif profit_r >= 0.75:
            lock_buffer = 0.12 * profit_r
        else:
            lock_buffer = 0.15 * profit_r

        if capital_ratio > 0.3:
            lock_buffer *= 0.7
        
        lock_buffer *= conservative_multiplier

        if side == "BUY":
            new_stop = entry_price + (peak - entry_price) * (1.0 - lock_buffer)
            ensure_profit = entry_price + (peak - entry_price) * 0.25
            new_stop = max(new_stop, ensure_profit)
        else:
            new_stop = entry_price - (entry_price - peak) * (1.0 - lock_buffer)
            ensure_profit = entry_price - (entry_price - peak) * 0.25
            new_stop = min(new_stop, ensure_profit)

        return SmartStopDecision(
            should_update=True,
            new_stop_price=round(new_stop, 8),
            stop_type=StopType.SMART_PROFIT.value,
            reason=f"Profit lock at R={profit_r:.2f}, protecting {(1-lock_buffer)*100:.1f}% gain",
            urgency=0.9,
            metadata={
                "profit_r": profit_r,
                "lock_buffer_pct": lock_buffer * 100,
                "peak": peak,
            }
        )

    def _trailing_stop(
        self,
        side: str,
        entry_price: float,
        current_price: float,
        current_stop: float,
        atr: float,
        profit_r: float,
        giveback_r: float,
        peak: float,
        volatility_factor: float,
        conservative_multiplier: float,
    ) -> SmartStopDecision:
        """Trailing stop dinamico basado en ATR y profit."""
        
        base_trail_multiplier = self.config.trailing_atr_multiplier
        
        if profit_r >= 1.5:
            trail_multiplier = base_trail_multiplier * 0.6
        elif profit_r >= 1.0:
            trail_multiplier = base_trail_multiplier * 0.75
        elif profit_r >= 0.75:
            trail_multiplier = base_trail_multiplier * 0.85
        else:
            trail_multiplier = base_trail_multiplier

        trail_multiplier *= volatility_factor * conservative_multiplier

        trail_distance = atr * trail_multiplier

        if side == "BUY":
            new_stop = peak - trail_distance
            if new_stop <= current_stop:
                return SmartStopDecision(
                    False, None, StopType.TRAILING.value,
                    f"Trailing stop higher than current: new={new_stop:.6f} vs current={current_stop:.6f}",
                    0.0
                )
            if profit_r > 0.1:
                min_profit_stop = entry_price + max(atr * profit_r * 0.3, atr * 0.5)
                new_stop = max(new_stop, min_profit_stop)
        else:
            new_stop = peak + trail_distance
            if new_stop >= current_stop:
                return SmartStopDecision(
                    False, None, StopType.TRAILING.value,
                    f"Trailing stop lower than current: new={new_stop:.6f} vs current={current_stop:.6f}",
                    0.0
                )
            if profit_r > 0.1:
                min_profit_stop = entry_price - max(atr * profit_r * 0.3, atr * 0.5)
                new_stop = min(new_stop, min_profit_stop)

        urgency = min(0.7 + giveback_r * 0.3, 1.0) if giveback_r > 0 else 0.5

        return SmartStopDecision(
            should_update=True,
            new_stop_price=round(new_stop, 8),
            stop_type=StopType.TRAILING.value,
            reason=f"Trailing stop at {trail_multiplier:.2f}x ATR (profit R={profit_r:.2f})",
            urgency=urgency,
            metadata={
                "trail_distance": trail_distance,
                "trail_multiplier": trail_multiplier,
                "profit_r": profit_r,
                "giveback_r": giveback_r,
            }
        )

    def _break_even_stop(
        self,
        side: str,
        entry_price: float,
        current_price: float,
        current_stop: float,
        atr: float,
        profit_r: float,
        conservative_multiplier: float,
    ) -> SmartStopDecision:
        """Mueve stop a break-even con pequeño buffer."""

        buffer_pct = self.config.break_even_buffer_profit_r * conservative_multiplier * 0.01
        buffer_pct = max(buffer_pct, 0.001)
        
        if side == "BUY":
            new_stop = entry_price * (1 + buffer_pct)
            if current_stop >= entry_price:
                return SmartStopDecision(
                    False, None, StopType.BREAK_EVEN.value,
                    "Already at or above break-even", 0.0
                )
            if new_stop <= current_stop:
                return SmartStopDecision(
                    False, None, StopType.BREAK_EVEN.value,
                    f"Break-even not improving stop: new={new_stop:.6f} vs current={current_stop:.6f}", 0.0
                )
        else:
            new_stop = entry_price * (1 - buffer_pct)
            if current_stop <= entry_price:
                return SmartStopDecision(
                    False, None, StopType.BREAK_EVEN.value,
                    "Already at or below break-even", 0.0
                )
            if new_stop >= current_stop:
                return SmartStopDecision(
                    False, None, StopType.BREAK_EVEN.value,
                    f"Break-even not improving stop: new={new_stop:.6f} vs current={current_stop:.6f}", 0.0
                )

        return SmartStopDecision(
            should_update=True,
            new_stop_price=round(new_stop, 8),
            stop_type=StopType.BREAK_EVEN.value,
            reason=f"Break-even stop with {buffer_pct*100:.2f}% buffer",
            urgency=0.6,
            metadata={"buffer_pct": buffer_pct * 100}
        )

    def _time_decay_stop(
        self,
        side: str,
        entry_price: float,
        current_price: float,
        current_stop: float,
        atr: float,
        time_factor: float,
        profit_r: float,
    ) -> SmartStopDecision:
        """Aprieta el stop basado en tiempo transcurrido."""

        decay_factor = min(time_factor * self.config.time_decay_aggression, 0.5)
        
        if side == "BUY":
            distance_to_entry = current_price - entry_price
            current_risk = current_price - current_stop
            decay_amount = current_risk * decay_factor
            new_stop = current_stop + decay_amount
        else:
            distance_to_entry = entry_price - current_price
            current_risk = current_stop - current_price
            decay_amount = current_risk * decay_factor
            new_stop = current_stop - decay_amount

        return SmartStopDecision(
            should_update=True,
            new_stop_price=round(new_stop, 8),
            stop_type=StopType.DYNAMIC_ATR.value,
            reason=f"Time decay stop: {decay_factor*100:.1f}% reduction after {time_factor:.1f} decay factor",
            urgency=0.4,
            metadata={"time_factor": time_factor, "decay_pct": decay_factor * 100}
        )

    def _dynamic_atr_stop(
        self,
        side: str,
        entry_price: float,
        current_price: float,
        current_stop: float,
        atr: float,
        volatility_factor: float,
        profit_r: float,
    ) -> SmartStopDecision:
        """Stop dinamico basado en ATR ajustado por volatilidad."""

        atr_multiplier = self.config.initial_stop_atr_multiplier * volatility_factor
        
        atr_multiplier = min(atr_multiplier, self.config.max_stop_distance_atr)

        if side == "BUY":
            ideal_stop = current_price - (atr * atr_multiplier)
            if profit_r > 0.1:
                min_profit_stop = entry_price + (atr * profit_r * 0.2)
                ideal_stop = max(ideal_stop, min_profit_stop)
        else:
            ideal_stop = current_price + (atr * atr_multiplier)
            if profit_r > 0.1:
                min_profit_stop = entry_price - (atr * profit_r * 0.2)
                ideal_stop = min(ideal_stop, min_profit_stop)

        if side == "BUY" and ideal_stop <= current_stop:
            return SmartStopDecision(False, None, StopType.DYNAMIC_ATR.value, "ATR stop not tightened", 0.0)
        if side == "SELL" and ideal_stop >= current_stop:
            return SmartStopDecision(False, None, StopType.DYNAMIC_ATR.value, "ATR stop not tightened", 0.0)

        return SmartStopDecision(
            should_update=True,
            new_stop_price=round(ideal_stop, 8),
            stop_type=StopType.DYNAMIC_ATR.value,
            reason=f"Dynamic ATR stop at {atr_multiplier:.2f}x ATR",
            urgency=0.3,
            metadata={"atr_multiplier": atr_multiplier}
        )

    def _calculate_volatility_factor(
        self,
        market_data: dict[str, Any] | None,
        atr: float,
        price: float,
    ) -> float:
        """
        Calcula factor de ajuste por volatilidad.
        
        Higher volatility = wider stops (more room for price movement)
        Lower volatility = tighter stops (less room needed)
        """
        if not self.config.volatility_adjustment:
            return 1.0

        atr_ratio = atr / max(price, 1e-9)

        # Low volatility (tight stops - market is calm)
        if atr_ratio < 0.003:
            return 0.85
        elif atr_ratio < 0.008:
            return 0.95
        elif atr_ratio < 0.015:
            return 1.0
        # High volatility (wider stops - market is volatile)
        elif atr_ratio < 0.025:
            return 1.15
        else:
            return 1.30

    def _calculate_time_factor(self, position_id: str) -> float:
        """Calcula factor de decay temporal."""
        if not self.config.time_decay_enabled:
            return 0.0

        entry_time = self._position_entry_times.get(position_id)
        if not entry_time:
            return 0.0

        elapsed_minutes = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60.0
        
        if elapsed_minutes < self.config.time_decay_start_minutes:
            return 0.0

        decay_factor = (elapsed_minutes - self.config.time_decay_start_minutes) / 60.0
        return min(decay_factor, 2.0)

    def _calculate_urgency(
        self,
        profit_r: float,
        giveback_r: float,
        time_factor: float,
        volatility_factor: float,
    ) -> float:
        """Calcula la urgencia de actualizar el stop."""
        urgency = 0.0

        if profit_r > 0.5:
            urgency += 0.4

        if giveback_r > 0.3:
            urgency += 0.4

        if time_factor > 0.5:
            urgency += 0.2

        if volatility_factor > 1.2:
            urgency += 0.1

        return min(urgency, 1.0)

    def should_emergency_exit(
        self,
        position_id: str,
        current_price: float,
        entry_price: float,
        side: str,
        market_data: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        """Determina si se debe salir de emergencia."""

        peak = self._position_peaks.get(position_id, entry_price)
        
        if side == "BUY":
            max_adverse = peak - current_price
            initial_peak = entry_price
        else:
            max_adverse = current_price - peak
            initial_peak = entry_price

        if market_data:
            frame5 = market_data.get("frame5")
            if frame5 is not None and hasattr(frame5, "iloc") and len(frame5) >= 3:
                last_3_closes = frame5["close"].iloc[-3:]
                if side == "BUY":
                    if all(last_3_closes.iloc[i] < last_3_closes.iloc[i-1] for i in range(1, len(last_3_closes))):
                        return True, "EMERGENCY_SELL_OFF_3_BARS"
                else:
                    if all(last_3_closes.iloc[i] > last_3_closes.iloc[i-1] for i in range(1, len(last_3_closes))):
                        return True, "EMERGENCY_BUY_OFF_3_BARS"

        return False, ""

    def cleanup_position(self, position_id: str) -> None:
        """Limpia los datos de una posición cerrada."""
        self._position_peaks.pop(position_id, None)
        self._position_entry_times.pop(position_id, None)
        self._position_initial_stops.pop(position_id, None)
        self._position_high_watermarks.pop(position_id, None)
        self._position_current_stops.pop(position_id, None)