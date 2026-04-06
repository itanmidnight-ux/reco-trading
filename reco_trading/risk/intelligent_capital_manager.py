from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from enum import Enum
import math

logger = logging.getLogger(__name__)


class CapitalMode(Enum):
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    VERY_LARGE = "very_large"


class MarketCondition(Enum):
    CALM = "calm"
    NORMAL = "normal"
    VOLATILE = "volatile"
    CHAOTIC = "chaotic"


@dataclass
class AdaptiveProfile:
    name: str
    min_confidence: float
    max_confidence: float
    risk_per_trade: float
    max_trades_per_day: int
    adx_threshold: float
    volume_multiplier: float
    spread_threshold: float
    stop_atr_multiplier: float
    take_profit_multiplier: float
    cooldown_minutes: int
    max_drawdown_pct: float
    min_trade_size_usdt: float
    trailing_activation_r: float
    break_even_trigger_r: float
    profit_lock_trigger_r: float


ADAPTIVE_PROFILES: dict[CapitalMode, AdaptiveProfile] = {
    CapitalMode.MICRO: AdaptiveProfile(
        name="Micro Capital ($5-$20)",
        min_confidence=0.72,
        max_confidence=0.95,
        risk_per_trade=0.008,
        max_trades_per_day=50,
        adx_threshold=12.0,
        volume_multiplier=0.65,
        spread_threshold=0.002,
        stop_atr_multiplier=1.2,
        take_profit_multiplier=1.8,
        cooldown_minutes=2,
        max_drawdown_pct=3.0,
        min_trade_size_usdt=5.0,
        trailing_activation_r=0.20,
        break_even_trigger_r=0.15,
        profit_lock_trigger_r=0.35,
    ),
    CapitalMode.SMALL: AdaptiveProfile(
        name="Small Capital ($20-$100)",
        min_confidence=0.68,
        max_confidence=0.95,
        risk_per_trade=0.010,
        max_trades_per_day=80,
        adx_threshold=11.0,
        volume_multiplier=0.70,
        spread_threshold=0.0025,
        stop_atr_multiplier=1.3,
        take_profit_multiplier=2.0,
        cooldown_minutes=2,
        max_drawdown_pct=4.0,
        min_trade_size_usdt=10.0,
        trailing_activation_r=0.25,
        break_even_trigger_r=0.18,
        profit_lock_trigger_r=0.40,
    ),
    CapitalMode.MEDIUM: AdaptiveProfile(
        name="Medium Capital ($100-$500)",
        min_confidence=0.62,
        max_confidence=0.95,
        risk_per_trade=0.012,
        max_trades_per_day=120,
        adx_threshold=10.0,
        volume_multiplier=0.75,
        spread_threshold=0.003,
        stop_atr_multiplier=1.4,
        take_profit_multiplier=2.2,
        cooldown_minutes=2,
        max_drawdown_pct=5.0,
        min_trade_size_usdt=15.0,
        trailing_activation_r=0.30,
        break_even_trigger_r=0.20,
        profit_lock_trigger_r=0.45,
    ),
    CapitalMode.LARGE: AdaptiveProfile(
        name="Large Capital ($500-$5000)",
        min_confidence=0.58,
        max_confidence=0.95,
        risk_per_trade=0.015,
        max_trades_per_day=200,
        adx_threshold=9.0,
        volume_multiplier=0.80,
        spread_threshold=0.0035,
        stop_atr_multiplier=1.5,
        take_profit_multiplier=2.5,
        cooldown_minutes=1,
        max_drawdown_pct=6.0,
        min_trade_size_usdt=20.0,
        trailing_activation_r=0.35,
        break_even_trigger_r=0.22,
        profit_lock_trigger_r=0.50,
    ),
    CapitalMode.VERY_LARGE: AdaptiveProfile(
        name="Very Large Capital ($5000+)",
        min_confidence=0.55,
        max_confidence=0.95,
        risk_per_trade=0.018,
        max_trades_per_day=500,
        adx_threshold=8.0,
        volume_multiplier=0.85,
        spread_threshold=0.004,
        stop_atr_multiplier=1.6,
        take_profit_multiplier=2.8,
        cooldown_minutes=1,
        max_drawdown_pct=7.0,
        min_trade_size_usdt=25.0,
        trailing_activation_r=0.40,
        break_even_trigger_r=0.25,
        profit_lock_trigger_r=0.55,
    ),
}


class IntelligentCapitalManager:
    """
    Gestor inteligente de capital que ajusta parametros dinamicamente
    segun el tamaño del capital, condiciones de mercado y rendimiento historico.
    
    - Capital pequeño: filtros más estrictos, menos riesgo por trade
    - Capital grande: filtros más relajados, más oportunidades
    - Adapta continuamente basado en rendimiento reciente
    """

    def __init__(
        self,
        initial_capital: float = 100.0,
        aggressive_mode: bool = False,
        preserve_capital: bool = True,
    ):
        self.initial_capital = max(initial_capital, 5.0)
        self.current_capital = self.initial_capital
        self.aggressive_mode = aggressive_mode
        self.preserve_capital = preserve_capital
        
        self.capital_mode = self._determine_mode(self.initial_capital)
        self.base_profile = ADAPTIVE_PROFILES[self.capital_mode]
        
        self._recent_pnls: list[float] = []
        self._recent_trades: int = 0
        self._win_streak: int = 0
        self._loss_streak: int = 0
        self._max_consecutive_wins: int = 0
        self._max_consecutive_losses: int = 0
        self._peak_capital = self.initial_capital
        self._current_drawdown: float = 0.0
        self._daily_trades: int = 0
        self._last_day: datetime = datetime.now(timezone.utc).date()
        self._market_condition = MarketCondition.NORMAL
        self._volatility_adjustments: dict[str, float] = {}
        self._performance_history: list[dict[str, Any]] = []
        self._last_adjustment_time = datetime.now(timezone.utc)
        
        logger.info(
            f"IntelligentCapitalManager initialized: mode={self.capital_mode.value} "
            f"capital=${self.initial_capital:.2f} profile={self.base_profile.name}"
        )

    def _determine_mode(self, capital: float) -> CapitalMode:
        """Determina el modo de capital basado en el monto."""
        if capital < 20:
            return CapitalMode.MICRO
        elif capital < 100:
            return CapitalMode.SMALL
        elif capital < 500:
            return CapitalMode.MEDIUM
        elif capital < 5000:
            return CapitalMode.LARGE
        else:
            return CapitalMode.VERY_LARGE

    def update_capital(self, current_balance: float) -> None:
        """Actualiza el capital y recalcula modo si es necesario."""
        self.current_capital = max(current_balance, 1.0)
        
        if self.current_capital > self._peak_capital:
            self._peak_capital = self.current_capital
        
        self._current_drawdown = max(
            0.0,
            (self._peak_capital - self.current_capital) / max(self._peak_capital, 1.0)
        )
        
        new_mode = self._determine_mode(self.current_capital)
        if new_mode != self.capital_mode:
            old_mode = self.capital_mode
            self.capital_mode = new_mode
            self.base_profile = ADAPTIVE_PROFILES[new_mode]
            logger.info(
                f"Capital mode changed: {old_mode.value} -> {new_mode.value} "
                f"(capital: ${self.current_capital:.2f})"
            )

    def record_trade(self, pnl: float, win: bool) -> None:
        """Registra resultado de trade para adaptación."""
        self._recent_pnls.append(pnl)
        if len(self._recent_pnls) > 100:
            self._recent_pnls = self._recent_pnls[-100:]
        
        self._recent_trades += 1
        
        if win:
            self._win_streak += 1
            self._loss_streak = 0
            self._max_consecutive_wins = max(self._max_consecutive_wins, self._win_streak)
        else:
            self._loss_streak += 1
            self._win_streak = 0
            self._max_consecutive_losses = max(self._max_consecutive_losses, self._loss_streak)
        
        self._daily_trades += 1

    def update_market_condition(
        self,
        volatility_ratio: float,
        atr_pct: float,
        spread_ratio: float,
    ) -> MarketCondition:
        """Actualiza condición de mercado detectada."""
        if volatility_ratio > 0.04 or atr_pct > 4.0 or spread_ratio > 0.006:
            self._market_condition = MarketCondition.CHAOTIC
        elif volatility_ratio > 0.025 or atr_pct > 2.5 or spread_ratio > 0.004:
            self._market_condition = MarketCondition.VOLATILE
        elif volatility_ratio > 0.012 or atr_pct > 1.5 or spread_ratio > 0.002:
            self._market_condition = MarketCondition.NORMAL
        else:
            self._market_condition = MarketCondition.CALM
        
        return self._market_condition

    def get_effective_parameters(self) -> dict[str, Any]:
        """
        Calcula parametros efectivos basados en:
        - Perfil base segun capital
        - Condición de mercado actual
        - Streak de wins/losses
        - Drawdown actual
        - Modo agresivo/conservador
        """
        profile = self.base_profile
        params = {
            "min_confidence": profile.min_confidence,
            "risk_per_trade": profile.risk_per_trade,
            "max_trades_per_day": profile.max_trades_per_day,
            "adx_threshold": profile.adx_threshold,
            "volume_multiplier": profile.volume_multiplier,
            "spread_threshold": profile.spread_threshold,
            "stop_atr_multiplier": profile.stop_atr_multiplier,
            "take_profit_multiplier": profile.take_profit_multiplier,
            "cooldown_minutes": profile.cooldown_minutes,
            "max_drawdown_pct": profile.max_drawdown_pct,
            "trailing_activation_r": profile.trailing_activation_r,
            "break_even_trigger_r": profile.break_even_trigger_r,
            "profit_lock_trigger_r": profile.profit_lock_trigger_r,
        }

        if self._market_condition == MarketCondition.CHAOTIC:
            params["min_confidence"] = min(params["min_confidence"] + 0.10, 0.95)
            params["risk_per_trade"] *= 0.5
            params["max_trades_per_day"] = max(params["max_trades_per_day"] // 4, 10)
            params["adx_threshold"] += 5.0
        elif self._market_condition == MarketCondition.VOLATILE:
            params["min_confidence"] = min(params["min_confidence"] + 0.05, 0.95)
            params["risk_per_trade"] *= 0.7
            params["max_trades_per_day"] = max(params["max_trades_per_day"] // 2, 20)
            params["adx_threshold"] += 2.0
        elif self._market_condition == MarketCondition.CALM:
            params["min_confidence"] = max(params["min_confidence"] - 0.03, 0.50)
            params["risk_per_trade"] *= 1.1
            params["adx_threshold"] = max(params["adx_threshold"] - 1.0, 6.0)

        if self.preserve_capital and self.current_capital < self.initial_capital * 0.8:
            preservation_factor = self.current_capital / (self.initial_capital * 0.8)
            params["min_confidence"] = min(params["min_confidence"] + (1 - preservation_factor) * 0.15, 0.95)
            params["risk_per_trade"] *= preservation_factor
            params["max_trades_per_day"] = max(int(params["max_trades_per_day"] * preservation_factor), 10)

        if self._loss_streak >= 3:
            streak_penalty = min(self._loss_streak * 0.03, 0.15)
            params["min_confidence"] = min(params["min_confidence"] + streak_penalty, 0.95)
            params["risk_per_trade"] *= max(1 - streak_penalty, 0.5)
        elif self._win_streak >= 5:
            streak_bonus = min(self._win_streak * 0.01, 0.05)
            params["min_confidence"] = max(params["min_confidence"] - streak_bonus, 0.50)
            params["risk_per_trade"] *= min(1 + streak_bonus, 1.3)

        if self._current_drawdown > profile.max_drawdown_pct * 0.5:
            dd_factor = self._current_drawdown / profile.max_drawdown_pct
            params["min_confidence"] = min(params["min_confidence"] + dd_factor * 0.05, 0.95)
            params["risk_per_trade"] *= max(1 - dd_factor * 0.3, 0.3)

        if self.aggressive_mode:
            params["min_confidence"] = max(params["min_confidence"] - 0.05, 0.50)
            params["risk_per_trade"] *= 1.2
            params["max_trades_per_day"] = int(params["max_trades_per_day"] * 1.5)

        if self.capital_mode == CapitalMode.MICRO:
            params["min_confidence"] = max(params["min_confidence"], 0.65)
            params["risk_per_trade"] = min(params["risk_per_trade"], 0.015)
            if self._loss_streak >= 2:
                params["min_confidence"] = min(params["min_confidence"] + 0.10, 0.95)

        params["min_confidence"] = round(params["min_confidence"], 4)
        params["risk_per_trade"] = round(params["risk_per_trade"], 4)
        params["adx_threshold"] = round(params["adx_threshold"], 2)
        params["volume_multiplier"] = round(params["volume_multiplier"], 3)
        
        return params

    def can_open_trade(
        self,
        current_trades: int,
        current_drawdown: float,
        consecutive_losses: int,
    ) -> tuple[bool, str]:
        """Verifica si se puede abrir un nuevo trade."""
        params = self.get_effective_parameters()
        
        now = datetime.now(timezone.utc)
        if now.date() != self._last_day:
            self._daily_trades = 0
            self._last_day = now.date()
        
        if self._daily_trades >= params["max_trades_per_day"]:
            return False, f"MAX_DAILY_TRADES ({params['max_trades_per_day']})"
        
        if current_drawdown >= params["max_drawdown_pct"] / 100:
            return False, f"MAX_DRAWDOWN ({params['max_drawdown_pct']}%)"
        
        if consecutive_losses >= 5:
            return False, f"CONSECUTIVE_LOSSES ({consecutive_losses})"
        
        if self._market_condition == MarketCondition.CHAOTIC and consecutive_losses >= 2:
            return False, "CHAOTIC_MARKET_PAUSE"
        
        return True, "OK"

    def calculate_position_size(
        self,
        equity: float,
        price: float,
        atr: float,
        confidence: float,
    ) -> float:
        """Calcula tamaño de posición inteligente."""
        params = self.get_effective_parameters()
        
        base_risk = params["risk_per_trade"]
        risk_per_trade = base_risk
        
        if confidence >= 0.85:
            risk_per_trade *= 1.2
        elif confidence >= 0.80:
            risk_per_trade *= 1.1
        elif confidence < 0.70:
            risk_per_trade *= 0.8
        elif confidence < 0.65:
            risk_per_trade *= 0.6
        
        if self.capital_mode == CapitalMode.MICRO:
            risk_per_trade = min(risk_per_trade, 0.012)
        elif self.capital_mode == CapitalMode.SMALL:
            risk_per_trade = min(risk_per_trade, 0.015)
        
        risk_amount = equity * risk_per_trade
        stop_distance = atr * params["stop_atr_multiplier"]
        stop_distance = max(stop_distance, price * 0.003)
        
        position_size = risk_amount / stop_distance
        position_size = position_size / price
        
        min_trade = params.get("min_trade_size_usdt", 5.0) / price
        max_trade_usdt = equity * params["max_drawdown_pct"] / 100
        max_trade = max_trade_usdt / price
        
        position_size = max(position_size, min_trade)
        position_size = min(position_size, max_trade)
        
        return round(position_size, 6)

    def get_signal_filters(self) -> dict[str, Any]:
        """Retorna filtros de señal actuales."""
        params = self.get_effective_parameters()
        
        return {
            "min_confidence": params["min_confidence"],
            "adx_threshold": params["adx_threshold"],
            "volume_filter_multiplier": params["volume_multiplier"],
            "max_spread_ratio": params["spread_threshold"],
            "require_trend_alignment": self.capital_mode in [CapitalMode.MICRO, CapitalMode.SMALL],
            "strict_volume_check": self.capital_mode == CapitalMode.MICRO,
            "require_multi_timeframe": self.capital_mode == CapitalMode.MICRO,
        }

    def get_stop_take_profit(
        self,
        entry_price: float,
        atr: float,
        side: str,
        confidence: float,
    ) -> dict[str, float]:
        """Calcula niveles de stop loss y take profit."""
        params = self.get_effective_parameters()
        
        stop_multiplier = params["stop_atr_multiplier"]
        tp_multiplier = params["take_profit_multiplier"]
        
        if confidence >= 0.85:
            stop_multiplier *= 0.85
            tp_multiplier *= 1.15
        elif confidence >= 0.80:
            stop_multiplier *= 0.9
            tp_multiplier *= 1.1
        elif confidence < 0.70:
            stop_multiplier *= 1.1
            tp_multiplier *= 0.9
        
        if side == "BUY":
            stop_loss = entry_price - (atr * stop_multiplier)
            take_profit = entry_price + (atr * tp_multiplier)
        else:
            stop_loss = entry_price + (atr * stop_multiplier)
            take_profit = entry_price - (atr * tp_multiplier)
        
        return {
            "stop_loss": round(stop_loss, 8),
            "take_profit": round(take_profit, 8),
            "stop_distance_pct": abs(entry_price - stop_loss) / entry_price * 100,
            "tp_distance_pct": abs(take_profit - entry_price) / entry_price * 100,
            "risk_reward_ratio": tp_multiplier / stop_multiplier,
        }

    def get_status(self) -> dict[str, Any]:
        """Retorna estado actual del gestor."""
        params = self.get_effective_parameters()
        return {
            "capital_mode": self.capital_mode.value,
            "capital_mode_name": self.base_profile.name,
            "current_capital": round(self.current_capital, 2),
            "initial_capital": round(self.initial_capital, 2),
            "peak_capital": round(self._peak_capital, 2),
            "current_drawdown_pct": round(self._current_drawdown * 100, 2),
            "win_streak": self._win_streak,
            "loss_streak": self._loss_streak,
            "daily_trades": self._daily_trades,
            "market_condition": self._market_condition.value,
            "effective_params": {
                "min_confidence": params["min_confidence"],
                "risk_per_trade": params["risk_per_trade"],
                "max_trades_per_day": params["max_trades_per_day"],
                "adx_threshold": params["adx_threshold"],
            },
            "aggressive_mode": self.aggressive_mode,
            "preserve_capital": self.preserve_capital,
        }

    def set_aggressive_mode(self, enabled: bool) -> None:
        """Activa/desactiva modo agresivo."""
        self.aggressive_mode = enabled
        logger.info(f"Aggressive mode {'enabled' if enabled else 'disabled'}")

    def reset_daily(self) -> None:
        """Resetea contadores diarios."""
        now = datetime.now(timezone.utc)
        if now.date() != self._last_day:
            self._daily_trades = 0
            self._last_day = now.date()
            logger.info(f"Daily counters reset - {now.date()}")