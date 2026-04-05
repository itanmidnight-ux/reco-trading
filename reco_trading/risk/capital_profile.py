from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(slots=True)
class CapitalProfile:
    name: str
    min_equity: float
    max_equity: float | None
    reserve_ratio: float
    reserve_buffer_usdt: float
    risk_per_trade_fraction: float
    max_trade_balance_fraction: float
    min_confidence: float
    max_trades_per_day: int
    cooldown_minutes: int
    loss_pause_minutes: int
    loss_pause_after_consecutive: int
    max_spread_ratio: float
    min_expected_reward_risk: float
    min_operable_notional_buffer: float
    max_concurrent_trades: int
    entry_quality_floor: float
    size_multiplier: float = 1.0


class CapitalProfileManager:
    """Maps current account equity to a capital-aware execution profile.

    Optimized for small capital with NANO profile for $3-25.
    """

    def __init__(self) -> None:
        self._profiles = (
            CapitalProfile(
                name="NANO",
                min_equity=0.0,
                max_equity=25.0,
                reserve_ratio=0.40,
                reserve_buffer_usdt=1.0,
                risk_per_trade_fraction=0.04,
                max_trade_balance_fraction=0.10,
                min_confidence=0.65,
                max_trades_per_day=25,          # Límite de seguridad, no el objetivo mínimo
                cooldown_minutes=5,             # Cooldown corto para mayor frecuencia
                loss_pause_minutes=45,          # Reducido para recuperar actividad antes
                loss_pause_after_consecutive=2, # 2 pérdidas seguidas = pausa breve
                max_spread_ratio=0.0010,
                min_expected_reward_risk=2.5,   # RR más alcanzable con capital pequeño
                min_operable_notional_buffer=1.25,
                max_concurrent_trades=1,
                entry_quality_floor=0.70,
                size_multiplier=0.40,
            ),
            CapitalProfile(
                name="MICRO",
                min_equity=25.0,
                max_equity=50.0,
                reserve_ratio=0.35,
                reserve_buffer_usdt=3.0,
                risk_per_trade_fraction=0.03,
                max_trade_balance_fraction=0.12,
                min_confidence=0.62,
                max_trades_per_day=25,          # Límite de seguridad amplio
                cooldown_minutes=5,             # Cooldown reducido
                loss_pause_minutes=35,
                loss_pause_after_consecutive=2,
                max_spread_ratio=0.0015,
                min_expected_reward_risk=2.3,
                min_operable_notional_buffer=1.20,
                max_concurrent_trades=1,
                entry_quality_floor=0.68,
                size_multiplier=0.50,
            ),
            CapitalProfile(
                name="SMALL",
                min_equity=50.0,
                max_equity=100.0,
                reserve_ratio=0.25,
                reserve_buffer_usdt=5.0,
                risk_per_trade_fraction=0.025,
                max_trade_balance_fraction=0.15,
                min_confidence=0.58,
                max_trades_per_day=3,
                cooldown_minutes=8,
                loss_pause_minutes=45,
                loss_pause_after_consecutive=2,
                max_spread_ratio=0.0020,
                min_expected_reward_risk=2.5,
                min_operable_notional_buffer=1.15,
                max_concurrent_trades=1,
                entry_quality_floor=0.70,
                size_multiplier=0.65,
            ),
            CapitalProfile(
                name="MEDIUM",
                min_equity=100.0,
                max_equity=500.0,
                reserve_ratio=0.18,
                reserve_buffer_usdt=10.0,
                risk_per_trade_fraction=0.02,
                max_trade_balance_fraction=0.18,
                min_confidence=0.55,
                max_trades_per_day=4,
                cooldown_minutes=6,
                loss_pause_minutes=30,
                loss_pause_after_consecutive=3,
                max_spread_ratio=0.0025,
                min_expected_reward_risk=2.2,
                min_operable_notional_buffer=1.12,
                max_concurrent_trades=1,
                entry_quality_floor=0.68,
                size_multiplier=0.75,
            ),
            CapitalProfile(
                name="LARGE",
                min_equity=500.0,
                max_equity=1000.0,
                reserve_ratio=0.12,
                reserve_buffer_usdt=25.0,
                risk_per_trade_fraction=0.015,
                max_trade_balance_fraction=0.20,
                min_confidence=0.52,
                max_trades_per_day=5,
                cooldown_minutes=5,
                loss_pause_minutes=25,
                loss_pause_after_consecutive=3,
                max_spread_ratio=0.0030,
                min_expected_reward_risk=2.0,
                min_operable_notional_buffer=1.08,
                max_concurrent_trades=1,
                entry_quality_floor=0.65,
                size_multiplier=0.85,
            ),
            CapitalProfile(
                name="PREMIUM",
                min_equity=1000.0,
                max_equity=None,
                reserve_ratio=0.08,
                reserve_buffer_usdt=50.0,
                risk_per_trade_fraction=0.012,
                max_trade_balance_fraction=0.25,
                min_confidence=0.50,
                max_trades_per_day=6,
                cooldown_minutes=4,
                loss_pause_minutes=20,
                loss_pause_after_consecutive=4,
                max_spread_ratio=0.0040,
                min_expected_reward_risk=1.8,
                min_operable_notional_buffer=1.05,
                max_concurrent_trades=1,
                entry_quality_floor=0.62,
                size_multiplier=1.0,
            ),
        )

    def select(self, equity: float) -> CapitalProfile:
        safe_equity = max(float(equity), 0.0)
        for profile in self._profiles:
            upper_ok = profile.max_equity is None or safe_equity < profile.max_equity
            if safe_equity >= profile.min_equity and upper_ok:
                if profile.name == "NANO":
                    return replace(self._profiles[1], min_equity=0.0, min_confidence=max(self._profiles[1].min_confidence, 0.70))
                if profile.name == "PREMIUM":
                    return replace(self._profiles[4], max_equity=None, max_trades_per_day=max(self._profiles[4].max_trades_per_day, 10))
                return profile
        return self._profiles[-1]

    @staticmethod
    def operable_capital(equity: float, profile: CapitalProfile, capital_limit: float | None = None) -> float:
        safe_equity = max(float(equity), 0.0)
        if capital_limit is not None and capital_limit > 0:
            safe_equity = min(safe_equity, float(capital_limit))
        reserved = max(safe_equity * float(profile.reserve_ratio), float(profile.reserve_buffer_usdt))
        return max(safe_equity - reserved, 0.0)
